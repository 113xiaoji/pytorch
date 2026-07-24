# Owner(s): ["module: dynamo"]
import abc
import functools
import inspect
import os
import subprocess
import sys
import textwrap
import unittest
import weakref

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._C._dynamo import guards
from torch._dynamo.convert_frame import GlobalStateGuard
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch.testing._internal.common_utils import set_default_dtype


RootGuardManager = guards.RootGuardManager
DictGuardManager = guards.DictGuardManager
GetAttrGuardAccessor = guards.GetAttrGuardAccessor
GetItemGuardAccessor = guards.GetItemGuardAccessor
TypeGuardAccessor = guards.TypeGuardAccessor
OBJECT_ALIASING = guards.OBJECT_ALIASING
install_object_aliasing_guard = guards.install_object_aliasing_guard
NO_TENSOR_ALIASING = guards.NO_TENSOR_ALIASING
install_no_tensor_aliasing_guard = guards.install_no_tensor_aliasing_guard


x = torch.tensor(4)
weakref_x = weakref.ref(x)

default_mgr_enum = torch._dynamo.guards.GuardManagerType.GUARD_MANAGER


class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y


global_pair = Pair(torch.randn(4), 1)


def id_type(x):
    return id(type(x))


def equals_match(x, expected):
    return x == expected


def equals_match_verbose_code_parts(expected):
    return [f"x == {expected}"]


def ge_match(x, expected):
    return x >= expected


def ge_match_verbose_code_parts(expected):
    return f"expected >= {expected}"


def less_match(x, expected):
    return x < expected


def less_match_verbose_code_parts(expected):
    return [f"expected < {expected}"]


class GuardManagerTests(torch._dynamo.test_case.TestCase):
    def test_global_state_guard(self):
        root = RootGuardManager()
        guard = guards.GLOBAL_STATE(root, ["global_state_check"])
        self.assertTrue(guard(None))
        with set_default_dtype(torch.double):
            self.assertFalse(guard(None))
            self.assertExpectedInline(
                str(guard.check_verbose(None)),
                """\
GuardDebugInfo(
result=0,
verbose_code_parts=['GLOBAL_STATE changed: default_dtype '],
num_guards_executed=0)
""",
            )
        self.assertTrue(guard(None))
        self.assertTrue(guard.check_verbose(None).result)
        _orig = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(not _orig)
            self.assertFalse(guard(None))
            self.assertExpectedInline(
                str(guard.check_verbose(None)),
                """\
GuardDebugInfo(
result=0,
verbose_code_parts=['GLOBAL_STATE changed: deterministic_algorithms '],
num_guards_executed=0)
""",
            )
        finally:
            torch.use_deterministic_algorithms(_orig)
        self.assertTrue(guard(None))
        self.assertTrue(guard.check_verbose(None).result)

    def test_global_state_reason(self):
        with torch.enable_grad():
            guards = GlobalStateGuard()
        with torch.no_grad():
            self.assertIs(guards.check(), False)
            self.assertEqual(guards.reason(), "grad_mode ")

    def test_python_lambda_leaf_guard(self):
        root = RootGuardManager()
        const_guard = guards.LAMBDA_GUARD(
            root,
            functools.partial(equals_match, expected=5),
            equals_match_verbose_code_parts(5),
        )
        self.assertTrue(const_guard(5))
        self.assertFalse(const_guard(4))
        self.assertFalse(const_guard("foo"))

    def test_type_guard(self):
        root = RootGuardManager()
        foo = 4
        guard = guards.TYPE_MATCH(root, id_type(foo), ["type(x) == int"])

        self.assertTrue(guard(5))
        self.assertTrue(guard(4))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.TYPE_MATCH(root, id_type(foo), ["type(x) == dict"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)

        guard = guards.TYPE_MATCH(root, id_type(foo), ["type(x) == Foo"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

    def test_id_guard(self):
        root = RootGuardManager()
        foo = 4
        guard = guards.ID_MATCH(root, id(foo), ["id(x) == id(foo)"])

        self.assertTrue(guard(foo))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.ID_MATCH(root, id(foo), ["id(x) == id(foo)"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard({"a": 1}))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))

    def test_equals_guard(self):
        root = RootGuardManager()
        foo = 4
        guard = guards.EQUALS_MATCH(root, foo, ["x == 4"])

        self.assertTrue(guard(4))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        # tuple
        foo = (1, 2, 3)
        guard = guards.EQUALS_MATCH(root, foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard((1, 2, 3)))
        self.assertFalse(guard((1, 2, 3, 4)))
        self.assertFalse(guard({}))

        # list
        foo = [1, 2, 3]
        guard = guards.EQUALS_MATCH(root, foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard([1, 2, 3]))
        self.assertFalse(guard([1, 2, 3, 4]))

        # type
        foo = int
        guard = guards.EQUALS_MATCH(root, foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard(int))
        self.assertFalse(guard(float))

    def test_default_device_guard(self):
        root = RootGuardManager()
        foo = 1
        guard = guards.DEFAULT_DEVICE(root, ["cpu device"])
        self.assertTrue(guard(foo))

        try:
            torch.set_default_device("cuda")
            self.assertFalse(guard(foo))
        finally:
            torch.set_default_device(None)

    def test_length_check_guard(self):
        root = RootGuardManager()
        foo = [1, 2, 3]
        guard = guards.LENGTH_CHECK(root, len(foo), ["len(x) == len(foo)"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard([]))

    def test_no_hasattr_guard(self):
        root = RootGuardManager()

        class Bar:
            def __init__(self) -> None:
                self.bar = 2

        bar = Bar()

        class Foo:
            def __init__(self) -> None:
                self.foo = 2

        foo = Foo()

        guard = guards.NO_HASATTR(root, "foo", ["hasattr(x, 'foo') == False"])
        self.assertTrue(guard(bar))
        self.assertFalse(guard(foo))

    def test_tensor_aliasing_guard(self):
        guard_manager = RootGuardManager()

        a = torch.randn(3, 4)

        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        f_locals = Foo(a, a)

        x_guard_mgr = guard_manager.getattr_manager("x", "", a, default_mgr_enum)
        y_guard_mgr = guard_manager.getattr_manager("y", "", a, default_mgr_enum)
        install_object_aliasing_guard(x_guard_mgr, y_guard_mgr, ["x is y"])

        # Check structure
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertTrue(isinstance(x_guards[0], OBJECT_ALIASING))
        self.assertTrue(isinstance(y_guards[0], OBJECT_ALIASING))
        # Check that the two guards are the same object
        self.assertTrue(x_guards[0] is y_guards[0])

        f_locals_unaliased = Foo(torch.randn(3, 4), torch.randn(3, 4))
        self.assertEqual(len(x_guard_mgr.get_leaf_guards()), 1)
        self.assertEqual(len(y_guard_mgr.get_leaf_guards()), 1)
        self.assertTrue(guard_manager.check(f_locals))

        self.assertFalse(guard_manager.check(f_locals_unaliased))

    def test_dict_version_guard(self):
        root = RootGuardManager()
        foo = {"a": 1, "b": 2}
        guard = guards.DICT_VERSION(root, foo, ["x.version == foo.version"])

        self.assertTrue(guard(foo))
        self.assertFalse(guard(dict(foo)))
        foo["a"] = 2
        self.assertFalse(guard(foo))
        self.assertFalse(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({}))

    def test_dynamic_indices_guard(self):
        root = RootGuardManager()
        guard1 = guards.DYNAMIC_INDICES(root, set(), ["x.size(0) == y.size(0)"])
        guard2 = guards.DYNAMIC_INDICES(root, set({0, 1}), ["x.size(0) == y.size(0)"])

        x = torch.randn(4)
        self.assertTrue(guard1(x))
        self.assertTrue(guard2(x))

        x._dynamo_dynamic_indices = set({0})
        self.assertFalse(guard1(x))
        self.assertTrue(guard2(x))

        x._dynamo_dynamic_indices = set({2})
        self.assertFalse(guard1(x))
        self.assertFalse(guard2(x))

    def test_tensor_match_guard(self):
        guard_manager = RootGuardManager()
        x = torch.randn(4, 4)
        size = list(x.size())
        stride = list(x.stride())
        guard_manager.add_tensor_match_guard(
            x,
            size,
            stride,
            "x",
            ["check_tensor(x)"],
            type(x),
            torch._C._dispatch_keys(x),
        )
        self.assertTrue(guard_manager.check(x))
        self.assertTrue(guard_manager.check_verbose(x).result)
        self.assertTrue(guard_manager.check(torch.randn(4, 4)))
        self.assertTrue(guard_manager.check_verbose(torch.randn(4, 4)).result)
        self.assertFalse(guard_manager.check(x.t_()))

        x = torch.randn(4, 4)
        x.t_()
        debug_info = guard_manager.check_verbose(x)
        print(debug_info.verbose_code_parts[0])
        self.assertTrue(
            "tensor 'x' stride mismatch" in debug_info.verbose_code_parts[0]
        )

    def test_no_tensor_aliasing_guard(self):
        guard_manager = RootGuardManager()

        a = torch.randn(3, 4)

        class Foo:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        f_locals = Foo(a, a, a)

        x_guard_mgr = guard_manager.getattr_manager("x", "", a, default_mgr_enum)
        y_guard_mgr = guard_manager.getattr_manager("y", "", a, default_mgr_enum)
        z_guard_mgr = guard_manager.getattr_manager("z", "", a, default_mgr_enum)
        install_no_tensor_aliasing_guard(
            [x_guard_mgr, y_guard_mgr, z_guard_mgr],
            ["x", "y", "z"],
            ["no_aliasing(x, y, z)"],
        )

        # Check structure
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        z_guards = z_guard_mgr.get_leaf_guards()
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertEqual(len(z_guards), 1)
        self.assertTrue(isinstance(x_guards[0], NO_TENSOR_ALIASING))
        self.assertTrue(isinstance(y_guards[0], NO_TENSOR_ALIASING))
        self.assertTrue(isinstance(z_guards[0], NO_TENSOR_ALIASING))
        # Check that the two guards are the same object
        self.assertTrue(x_guards[0] is y_guards[0] is z_guards[0])
        self.assertFalse(guard_manager.check(f_locals))
        self.assertFalse(guard_manager.check_verbose(f_locals).result)

        f_locals_unaliased = Foo(
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        )
        self.assertTrue(guard_manager.check(f_locals_unaliased))
        self.assertTrue(guard_manager.check_verbose(f_locals_unaliased).result)
        # Check that hash map is cleared.
        self.assertTrue(guard_manager.check(f_locals_unaliased))

        f_locals_unaliased = Foo(
            a,
            torch.randn(3, 4),
            a,
        )
        self.assertFalse(guard_manager.check(f_locals_unaliased))
        self.assertFalse(guard_manager.check_verbose(f_locals_unaliased).result)

    def test_weakref_alive_guard(self):
        root = RootGuardManager()
        x = torch.rand(3, 4)
        weakref_x = weakref.ref(x)

        guard = guards.NOT_NONE(root, ["weakref_x is not None"])
        self.assertTrue(guard(weakref_x()))
        del x
        self.assertFalse(guard(weakref_x()))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_call_function_no_args_guard(self):
        root = RootGuardManager()
        x = torch.cuda.current_device()
        guard = guards.EQUALS_MATCH(root, x, [0])
        self.assertTrue(guard(0))
        self.assertFalse(guard(1))
        self.assertFalse(guard(2))

    def test_guard_manager_leaf_guard(self):
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(5), ["type(x) == int"])
        guard_manager.add_lambda_guard(
            functools.partial(ge_match, expected=5),
            ge_match_verbose_code_parts(expected=5),
        )
        guard_manager.add_lambda_guard(
            functools.partial(less_match, expected=10),
            less_match_verbose_code_parts(expected=10),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 3)
        self.assertEqual(len(guard_manager.get_accessors()), 0)
        self.assertTrue(guard_manager.check(6))
        self.assertFalse(guard_manager.check(4))
        self.assertFalse(guard_manager.check("foo"))

    def test_attr_guard_manager(self):
        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guard_manager.getattr_manager("x", "x", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo.x),
            equals_match_verbose_code_parts(foo.x),
        )
        guard_manager.getattr_manager("y", "y", 2, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo.y),
            equals_match_verbose_code_parts(foo.y),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetAttrGuardAccessor)
        )
        # Check leaf guards on child managers
        self.assertEqual(
            len(
                guard_manager.getattr_manager(
                    attr="x",
                    source="x",
                    example_value=None,
                    guard_manager_enum=default_mgr_enum,
                ).get_leaf_guards()
            ),
            1,
        )
        self.assertEqual(
            len(
                guard_manager.getattr_manager(
                    "y", "y", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check(Foo(3, 4)))
        self.assertFalse(guard_manager.check("foo"))

    def test_item_guard_manager(self):
        foo = [1, 2]
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guard_manager.getitem_manager(0, "", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo[0]),
            equals_match_verbose_code_parts(foo[0]),
        )
        guard_manager.getitem_manager(1, "", 2, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo[1]),
            equals_match_verbose_code_parts(foo[1]),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetItemGuardAccessor)
        )
        # Check leaf guards on child managers
        self.assertEqual(
            len(
                guard_manager.getitem_manager(
                    0, "", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )
        self.assertEqual(
            len(
                guard_manager.getitem_manager(
                    1, "", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check([3, 4]))
        self.assertFalse(guard_manager.check("foo"))

    def test_framelocals_accessor(self):
        foo = {
            "a": 1,
            "b": 2,
        }

        guards_manager = RootGuardManager()
        guards_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guards_manager.framelocals_manager(
            ("a", 0), "", 1, default_mgr_enum
        ).add_equals_match_guard(1, ["a == 1"])
        guards_manager.framelocals_manager(
            ("b", 1), "", 2, default_mgr_enum
        ).add_equals_match_guard(2, ["b == 2"])

        self.assertTrue(guards_manager.check(foo))
        self.assertFalse(guards_manager.check({"a": 1, "b": 3}))

    def test_framelocals_guard_e2e(self):
        def fn(x, y, z):
            return x + y + z[0]

        opt_fn = torch.compile(fn, backend="eager")

        ref = opt_fn(torch.ones(3), 2, {0: 1, 2: 3})
        with torch._dynamo.set_stance("fail_on_recompile"):
            res = opt_fn(torch.ones(3), 2, {0: 1, 2: 3})
        self.assertEqual(ref, res)

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 1)
        guard_str = str(c1[0].guard_manager)
        self.assertIn(
            "source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)",
            guard_str,
        )
        self.assertIn(
            "source=L['y'], accessed_by=FrameLocalsGuardAccessor(key='y', framelocals_idx=1)",
            guard_str,
        )
        self.assertIn(
            "source=L['z'], accessed_by=FrameLocalsGuardAccessor(key='z', framelocals_idx=2)",
            guard_str,
        )

    def test_dict_getitem_accessor(self):
        foo = {
            "a": 1,
            "b": 2,
        }

        guards_manager = RootGuardManager()
        guards_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guards_manager.dict_getitem_manager(
            "a", "", 1, default_mgr_enum
        ).add_equals_match_guard(1, ["a == 1"])
        guards_manager.dict_getitem_manager(
            "b", "", 2, default_mgr_enum
        ).add_equals_match_guard(2, ["b == 2"])

        self.assertTrue(guards_manager.check(foo))
        self.assertFalse(guards_manager.check({"a": 1, "b": 3}))

    def test_globals(self):
        global global_pair, Pair
        guard_manager = RootGuardManager()
        gpair_mgr = guard_manager.globals_dict_manager(
            globals(), "", None, default_mgr_enum
        ).getitem_manager("global_pair", "", global_pair, default_mgr_enum)

        gpair_mgr.add_lambda_guard(
            lambda x: isinstance(x, Pair)
            and isinstance(x.x, torch.Tensor)
            and isinstance(x.y, int),
            "global guard fail",
        )

        self.assertTrue(guard_manager.check(global_pair))
        global_pair.y = "foo"
        self.assertFalse(guard_manager.check(global_pair))

    def test_type_manager(self):
        guard_manager = RootGuardManager()

        class A:
            a = 4

        class B(A):
            def mul(self, x):
                super().mul(x)

        foo = B()
        f_locals = {"foo": foo}

        # len(type(foo).__mro__) == 2
        foo_mgr = guard_manager.getitem_manager("foo", "", foo, default_mgr_enum)
        type_manager = foo_mgr.type_manager("", type(foo), default_mgr_enum)
        self.assertTrue(isinstance(foo_mgr.get_accessors()[0], TypeGuardAccessor))
        mro_manager = type_manager.getattr_manager(
            "__mro__", "", type(foo).__mro__, default_mgr_enum
        )
        self.assertTrue(
            isinstance(type_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        mro_manager.add_length_check_guard(
            3,
            "Expected len(type(foo).__mro__) == 3",
        )

        # type(foo).__mro__[0].a = 4
        item_manager = mro_manager.getitem_manager(
            1, "", type(foo).__mro__[1], default_mgr_enum
        )
        self.assertTrue(
            isinstance(mro_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        attr_manager = item_manager.getattr_manager(
            "a", "", type(foo).__mro__[0].a, default_mgr_enum
        )
        self.assertTrue(
            isinstance(item_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        attr_manager.add_lambda_guard(
            lambda x: x == 4,
            "Expected value 4",
        )

        self.assertTrue(guard_manager.check(f_locals))

    def test_tuple_iterator_getitem(self):
        a = (1, 2, 3, 4, 5, 6)
        foo = iter(a)
        next(foo)  # foo points at index=1

        guard_manager = RootGuardManager()
        # Check a[3] which is tuple_iterator_getitem(foo, 2)
        guard_manager.add_tuple_iterator_length_guard(
            5, id_type(iter(())), ["len == 5"]
        )
        guard_manager.tuple_iterator_getitem_manager(
            2, "", foo, default_mgr_enum
        ).add_equals_match_guard(a[3], ["x==4"])

        # Check that type match works
        self.assertFalse(guard_manager.check(False))

        self.assertTrue(guard_manager.check(foo))

        # Check that index error fails gracefully
        b = (1, 2)
        b_foo = iter(b)
        self.assertFalse(guard_manager.check(b_foo))

    def test_global_weakref(self):
        guard_manager = RootGuardManager()
        globals_manager = guard_manager.globals_dict_manager(
            globals(), "", None, default_mgr_enum
        )
        weakref_manager = globals_manager.global_weakref_manager(
            "weakref_x", "", None, default_mgr_enum
        )

        weakref_manager.add_lambda_guard(
            lambda x: isinstance(x, torch.Tensor),
            "global weakref fail",
        )

        self.assertTrue(guard_manager.check(None))
        global x
        del x
        self.assertFalse(guard_manager.check(None))

    def test_lambda_manager(self):
        a = (1, 1, 3, 4, 5, 6)

        guard_manager = RootGuardManager()

        # Check that we can use the same accessor
        foo_mgr = guard_manager.lambda_manager(
            lambda x: x[2], "", None, default_mgr_enum
        )
        foo_mgr.add_lambda_guard(
            lambda x: x == 3,
            "Expected value 3",
        )
        self.assertTrue(guard_manager.check(a))

        # test that exception works
        guard_manager = RootGuardManager()

        def fn(x):
            raise AssertionError("Test")
            return x

        foo_mgr = guard_manager.lambda_manager(fn, "", None, default_mgr_enum)

        self.assertFalse(guard_manager.check(None))
        debug_info = guard_manager.check_verbose(None)
        self.assertFalse(debug_info.result)
        self.assertTrue("Test" in debug_info.verbose_code_parts[0])

    def test_dict_contains_guard(self):
        root = RootGuardManager()
        foo = {"a": 1, "b": 2}
        guard = guards.DICT_CONTAINS(root, True, "a", ["has a"])

        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertFalse(guard({}))

        guard = guards.DICT_CONTAINS(root, False, "c", ["not has c"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertTrue(guard({}))

    def test_dict_guard_manager(self):
        root = RootGuardManager()

        def nothing():
            pass

        f_locals = {
            "d": {"a": 1, nothing: {"z": 3}, 100: torch.randn(4)},
        }

        # its a getitem_manager just for f_locals. But the child guard manager
        # should be a DictGuardManager.
        dict_mgr = root.getitem_manager(
            "d",
            "",
            f_locals["d"],
            torch._dynamo.guards.GuardManagerType.DICT_GUARD_MANAGER,
        )
        self.assertTrue(isinstance(dict_mgr, DictGuardManager))

        self.assertTrue(root.check(f_locals))

        # Check that no one can add a leaf guard
        with self.assertRaises(RuntimeError):
            dict_mgr.add_id_match_guard(id_type(f_locals), "id match")

        # Check that no one can add an arbitrary accessor
        with self.assertRaises(RuntimeError):
            dict_mgr.getitem_manager("a", "", f_locals["d"]["a"])

        # Check that it fails with different length dict
        f_locals_prime = {
            "d": {"a": 1, "b": 2},
        }
        self.assertFalse(root.check(f_locals_prime))

        # Add key-value manager ("a" : 1)
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(0, "", "a", default_mgr_enum).add_equals_match_guard(
            "a",
            ["dict.keys()[0] == a"],
        )
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_value_manager(0, "", 1, default_mgr_enum).add_equals_match_guard(
            1, ["d[0] == 1"]
        )
        self.assertTrue(root.check(f_locals))

        # Add key-value manager (nothing : {"z" : 3})
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(1, "", nothing, default_mgr_enum).add_lambda_guard(
            lambda x: x is nothing, ["x is nothing"]
        )
        self.assertTrue(root.check(f_locals))
        value_mgr = dict_mgr.get_value_manager(
            1,
            "",
            f_locals["d"][nothing],
            torch._dynamo.guards.GuardManagerType.DICT_GUARD_MANAGER,
        )
        self.assertTrue(isinstance(value_mgr, DictGuardManager))
        self.assertTrue(root.check(f_locals))

        # Check structure
        # Check that we are only guarding on two keys. This is common in
        # LazyVariableTracker.
        self.assertEqual(len(dict_mgr.get_key_value_managers()), 2)

        f_locals["d"]["a"] = 2
        self.assertFalse(root.check(f_locals))
        self.assertFalse(root.check_verbose(f_locals).result)

        f_locals["d"]["a"] = 1
        self.assertTrue(root.check(f_locals))

        f_locals["d"].pop(100)
        # fails because of len check
        self.assertFalse(root.check(f_locals))

    def test_clone(self):
        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            root = guard_wrapper.root

            # Check full cloning works as expected
            cloned_root = root.clone_manager(lambda x: True)
            self.assertTrue(cloned_root.check(f_locals))
            f_locals["foo"] = [3, 4]
            self.assertFalse(cloned_root.check(f_locals))
            f_locals["foo"] = [2, 3]

            # Skip guarding on foo
            cloned_root = root.clone_manager(lambda x: "foo" not in x.get_source())
            f_locals["foo"] = [3, 4]
            # Original root should fail, but new root should pass because of
            # absence of guards on foo.
            self.assertFalse(root.check(f_locals))
            self.assertTrue(cloned_root.check(f_locals))

        class Bar:
            x = 4
            y = torch.randn(4)

        foo = [2, 3]
        bar = Bar()

        def fn(x, foo, bar):
            return x + foo[0] + bar.x * bar.y

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(x, foo, bar)

    def test_diff_guard_manager(self):
        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook
        counter = 0

        def hook(guard_wrapper, f_locals, builder):
            nonlocal counter
            root = guard_wrapper.root
            diff_guard_root = guard_wrapper.diff_guard_root

            # Check full cloning works as expected
            self.assertTrue(root.check(f_locals))
            self.assertTrue(diff_guard_root.check(f_locals))

            # Check that tensor guards run well
            old_tensor = f_locals["bar"].y
            f_locals["bar"].y = torch.randn(5)
            self.assertFalse(root.check(f_locals))
            self.assertFalse(diff_guard_root.check(f_locals))
            f_locals["bar"].y = old_tensor

            # Original root should fail on foo changes, but diff_guard_root
            # should pass because it does not have foo guards on counter = 0. On
            # counter = 1, it should pass because we have caused a recompile
            # because of foo, causing it to recompile on foo.
            f_locals["foo"] = [3, 3]
            self.assertFalse(root.check(f_locals))
            if counter == 0:
                self.assertTrue(diff_guard_root.check(f_locals))
            else:
                self.assertFalse(diff_guard_root.check(f_locals))
            counter += 1

        class Bar:
            def __init__(self):
                self.x = 4
                self.y = torch.randn(4)

        bar = Bar()

        def fn(x, foo, bar):
            return x + foo[0] + bar.x * bar.y

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            foo = (12.0, 13)
            opt_fn(x, foo, bar)

            foo = (10.0, 11)
            opt_fn(x, foo, bar)


class TypePropagationTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(skip_tensor_guards_with_matching_dict_tags=True)
    def test_basic_types(self):
        class Foo:
            def __init__(self):
                self.x = {"a": 2}
                self.y = torch.randn(4)
                self.z = {}

        foo = Foo()

        mod = torch.nn.Linear(4, 4)

        def fn(x):
            return x + foo.x["a"] + foo.y + mod(x)

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import AttrSource, DictGetItemSource, LocalSource

            foo_source = LocalSource("foo")
            foo_x_source = AttrSource(foo_source, "x")

            self.assertTrue(builder.get(foo_source) is foo)
            self.assertTrue(builder.get(foo_x_source) is foo.x)

            # Check types of foo.x
            foo_x_mgr = builder.get_guard_manager_from_source(foo_x_source)
            self.assertTrue(issubclass(foo_x_mgr.get_type_of_guarded_value(), dict))

            # Check types of foo.x["a"]
            foo_x_a_source = DictGetItemSource(foo_x_source, "a")
            foo_x_a_mgr = builder.get_guard_manager_from_source(foo_x_a_source)
            self.assertTrue(foo_x_a_mgr.is_guarded_value_immutable())

            # Check types of foo.y
            foo_y_source = AttrSource(foo_source, "y")
            foo_y_mgr = builder.get_guard_manager_from_source(foo_y_source)
            self.assertTrue(foo_y_mgr.is_guarded_value_immutable())

            # Check types of foo.z
            foo_z_source = AttrSource(foo_source, "z")
            foo_z_mgr = builder.get_guard_manager_from_source(foo_z_source)
            self.assertTrue(issubclass(foo_z_mgr.get_type_of_guarded_value(), dict))

            # Check types of mod
            mod_source = LocalSource("mod")
            mod_mgr = builder.get_guard_manager_from_source(mod_source)
            self.assertTrue(
                issubclass(mod_mgr.get_type_of_guarded_value(), torch.nn.Module)
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))


class DuplicateGuardTest(torch._dynamo.test_case.TestCase):
    def test_duplicate_guard(self):
        class Foo:
            def __init__(self):
                self.x = 4
                self.bar = 4

        foo = Foo()

        def fn(x):
            if hasattr(foo, "y"):
                x = torch.sin(x)
            if hasattr(foo, "y"):
                x = torch.sin(x)

            if hasattr(foo, "bar"):
                x = torch.cos(x)
            if hasattr(foo, "bar"):
                x = torch.cos(x)
            return x + foo.x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            guard_str = str(guard_wrapper)
            # One for tensor and one for y
            self.assertEqual(guard_str.count("NO_HASATTR"), 2)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))


class RecursiveDictTagTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        self._prev = torch._dynamo.config.use_recursive_dict_tags_for_guards
        torch._dynamo.config.use_recursive_dict_tags_for_guards = True

    def tearDown(self):
        torch._dynamo.config.use_recursive_dict_tags_for_guards = self._prev


class TagSafetyChecks(RecursiveDictTagTests):
    def setUp(self):
        self._prev = torch._dynamo.config.use_recursive_dict_tags_for_guards
        torch._dynamo.config.use_recursive_dict_tags_for_guards = True

    def tearDown(self):
        torch._dynamo.config.use_recursive_dict_tags_for_guards = self._prev

    def test_immutable_tag_safe(self):
        class Bar:
            pass

        class Foo:
            def __init__(self):
                self.a = Bar()
                self.b = torch.randn(4)
                self.c = 3
                self.d = (3, 4)
                self.e = (3, Bar())

        foo = Foo()

        def fn(x):
            if foo.a:
                x = torch.sin(x)
            x = x * foo.b + foo.c + foo.d[0] + foo.d[1] + foo.e[0]
            if foo.e[1]:
                x = torch.sin(x)
            return x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import AttrSource, LocalSource

            foo_source = LocalSource("foo")
            foo_mgr = builder.get_guard_manager_from_source(foo_source)
            for accessor in foo_mgr.get_accessors():
                if isinstance(accessor, GetAttrGuardAccessor):
                    self.assertTrue(
                        accessor.get_attr_name() in ("a", "b", "c", "d", "e")
                    )

            # Check types of foo.a
            foo_a_source = AttrSource(foo_source, "a")
            foo_a_mgr = builder.get_guard_manager_from_source(foo_a_source)
            self.assertFalse(foo_a_mgr.is_tag_safe())
            self.assertFalse(foo_a_mgr.is_tag_safe_root())

            # Check types of foo.b
            foo_b_source = AttrSource(foo_source, "b")
            foo_b_mgr = builder.get_guard_manager_from_source(foo_b_source)
            if torch._dynamo.config.skip_tensor_guards_with_matching_dict_tags:
                self.assertTrue(foo_b_mgr.is_tag_safe())
            else:
                self.assertFalse(foo_b_mgr.is_tag_safe())

            self.assertFalse(foo_b_mgr.is_tag_safe_root())

            # Check types of foo.c
            foo_c_source = AttrSource(foo_source, "c")
            foo_c_mgr = builder.get_guard_manager_from_source(foo_c_source)
            self.assertTrue(foo_c_mgr.is_tag_safe())
            self.assertFalse(foo_c_mgr.is_tag_safe_root())

            # Check types of foo.d
            foo_d_source = AttrSource(foo_source, "d")
            foo_d_mgr = builder.get_guard_manager_from_source(foo_d_source)
            self.assertTrue(foo_d_mgr.is_tag_safe())
            self.assertFalse(foo_d_mgr.is_tag_safe_root())

            # Check types of foo.e
            foo_e_source = AttrSource(foo_source, "e")
            foo_e_mgr = builder.get_guard_manager_from_source(foo_e_source)
            self.assertFalse(foo_e_mgr.is_tag_safe())
            self.assertFalse(foo_e_mgr.is_tag_safe_root())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))

    def test_dict_tag_safe(self):
        class Foo:
            def __init__(self):
                self.a = 4

        foo = Foo()
        terminal_dict = {
            "a": 1,
        }

        tag_safe_dict = {
            "const": 1,
            "tup": (2, 3),
            "nested_dict": terminal_dict,
        }

        tag_unsafe_dict = {
            "const": 1,
            "foo": foo,
        }

        outer_dict = {
            "safe": tag_safe_dict,
            "unsafe": tag_unsafe_dict,
            "terminal_dict": {"a": 1},
        }

        def fn(x):
            x = x + outer_dict["safe"]["const"]

            x = x + outer_dict["safe"]["tup"][0]
            x = x + outer_dict["safe"]["tup"][1]

            x = x + outer_dict["safe"]["nested_dict"]["a"]

            x = x + outer_dict["unsafe"]["const"]

            x = x + outer_dict["unsafe"]["foo"].a

            if outer_dict["terminal_dict"]:
                x = torch.sin(x)
            return x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import DictGetItemSource, LocalSource

            outer_source = LocalSource("outer_dict")

            # Check tagness of outer dict
            outer_mgr = builder.get_guard_manager_from_source(outer_source)
            self.assertFalse(outer_mgr.is_tag_safe())
            self.assertFalse(outer_mgr.is_tag_safe_root())

            # Check tagness of outer["safe"]
            outer_safe_source = DictGetItemSource(outer_source, "safe")
            outer_safe_mgr = builder.get_guard_manager_from_source(outer_safe_source)
            self.assertTrue(outer_safe_mgr.is_tag_safe())
            self.assertFalse(outer_safe_mgr.is_tag_safe_root())

            # Check tagness of outer["unsafe"]
            outer_unsafe_source = DictGetItemSource(outer_source, "unsafe")
            outer_unsafe_mgr = builder.get_guard_manager_from_source(
                outer_unsafe_source
            )
            self.assertFalse(outer_unsafe_mgr.is_tag_safe())
            self.assertFalse(outer_unsafe_mgr.is_tag_safe_root())

            # Check tagness of outer["terminal_dict"]
            outer_terminal_source = DictGetItemSource(outer_source, "terminal_dict")
            outer_terminal_mgr = builder.get_guard_manager_from_source(
                outer_terminal_source
            )
            self.assertTrue(outer_terminal_mgr.is_tag_safe())
            self.assertFalse(outer_terminal_mgr.is_tag_safe_root())

            # Check tagness of outer["safe"]["nested_dict"]
            outer_safe_nested_source = DictGetItemSource(
                outer_safe_source, "nested_dict"
            )
            outer_safe_nested_mgr = builder.get_guard_manager_from_source(
                outer_safe_nested_source
            )
            self.assertTrue(outer_safe_nested_mgr.is_tag_safe())
            # This should not be marked as a root
            self.assertFalse(outer_safe_nested_mgr.is_tag_safe_root())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))

    def test_nn_module_tag_safe(self):
        class Foo(torch.nn.Module):
            c = 2

            def __init__(self):
                super().__init__()
                self.a = 4

            def check(self, x):
                return True

            def forward(self, x):
                inspect.signature(self.check).parameters.items()
                return x + self.a + self.c

        foo = Foo()

        class Env(metaclass=abc.ABCMeta):  # noqa: B024
            pass

        class Baz(torch.nn.Module, Env):
            def __init__(self):
                super().__init__()
                self.foo = foo

            def forward(self, x):
                if "Foo" in str(type(self).__mro__):
                    x = torch.sin(x)
                return self.foo(x)

        baz = Baz()

        def fn(x):
            x = x + baz(x)
            return x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            baz_source = LocalSource("baz")

            # Check tagness of baz
            baz_mgr = builder.get_guard_manager_from_source(baz_source)
            self.assertTrue(baz_mgr.is_tag_safe())
            self.assertTrue(baz_mgr.is_tag_safe_root())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))

    def test_nn_module_tag_overridden_getattr_safe(self):
        class Baz(torch.nn.Module, metaclass=abc.ABCMeta):
            def __init__(self):
                super().__init__()
                self.norm = 2

            def __getattr__(self, key):
                if key == "a":
                    return 5
                return super().__getattr__(key)

            def forward(self, x):
                return x + self.a + self.norm

        baz = Baz()

        def fn(x):
            x = x + baz(x)
            return x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            baz_source = LocalSource("baz")

            # Check tagness of baz
            baz_mgr = builder.get_guard_manager_from_source(baz_source)
            self.assertTrue(baz_mgr.is_tag_safe())
            self.assertTrue(baz_mgr.is_tag_safe_root())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))


class RecursiveDictGuardTests(RecursiveDictTagTests):
    def test_disabling(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 4

            def forward(self, x):
                return x + self.a

        mod = Mod()
        mod_to_fail = Mod()

        def fn(x):
            return mod(x)

        x = torch.randn(4, 4)

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def basic_hook_test(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            mod_source = LocalSource("mod")

            # Check tagness of mod
            mod_mgr = builder.get_guard_manager_from_source(mod_source)
            self.assertTrue(mod_mgr.is_tag_safe())
            self.assertTrue(mod_mgr.is_tag_safe_root())
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            for _ in range(10):
                self.assertTrue(guard_wrapper.check({"mod": mod, "x": x}))
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            # Let the guard pass but dict matching fail, this should add new cached entry
            self.assertTrue(guard_wrapper.check({"mod": mod_to_fail, "x": x}))
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            # Let the guard fail, this should disable dict tag optimization as well
            mod_to_fail.a = 5
            self.assertFalse(guard_wrapper.check({"mod": mod_to_fail, "x": x}))
            self.assertTrue(mod_mgr.is_recursive_dict_tag_matching_disabled())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(basic_hook_test):
            opt_fn(x)

        # Test that dict tag matching failure leads to disable of dict tag optimization
        torch.compiler.reset()
        mod = Mod()
        mod_to_fail = Mod()

        def disable_on_dict_tag_match_failure(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            mod_source = LocalSource("mod")

            # Check tagness of mod
            mod_mgr = builder.get_guard_manager_from_source(mod_source)
            self.assertTrue(mod_mgr.is_tag_safe())
            self.assertTrue(mod_mgr.is_tag_safe_root())
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            for _ in range(10):
                self.assertTrue(guard_wrapper.check({"mod": mod, "x": x}))
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            # Change the mod attr to cause dict tag matching to fail, this still
            # get the guard pass. This should disable the dict tag optimization.
            mod.a = 5
            mod.a = 4
            self.assertTrue(guard_wrapper.check({"mod": mod, "x": x}))
            self.assertTrue(mod_mgr.is_recursive_dict_tag_matching_disabled())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(disable_on_dict_tag_match_failure):
            opt_fn(x)

        # Test that max size limit breach disables the dict tag optimization
        torch.compiler.reset()
        mod = Mod()
        mod_to_fail = Mod()

        def max_size_test(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            mod_source = LocalSource("mod")

            # Check tagness of mod
            mod_mgr = builder.get_guard_manager_from_source(mod_source)
            self.assertTrue(mod_mgr.is_tag_safe())
            self.assertTrue(mod_mgr.is_tag_safe_root())
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            for _ in range(10):
                self.assertTrue(guard_wrapper.check({"mod": mod, "x": x}))
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            # Let the guard pass but dict matching fail, since cache size is set
            # to 1, this would cause dict tag optimization to be disabled.
            self.assertTrue(guard_wrapper.check({"mod": mod_to_fail, "x": x}))
            self.assertTrue(mod_mgr.is_recursive_dict_tag_matching_disabled())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with torch._dynamo.config.patch(
            max_saved_pointers_for_recursive_dict_tags_check=1
        ):
            with install_guard_manager_testing_hook(max_size_test):
                opt_fn(x)


class GuardActualPartialFastPathTests(torch._dynamo.test_case.TestCase):
    def test_actual_partial_preserves_module_and_residual_guards(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            global_bias = torch.tensor(3.0)

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("scale", torch.tensor(2.0))
                    self.offsets = [1.0]

                def forward(self, x):
                    return x * self.scale + self.offsets[0] + global_bias

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )

            x = torch.ones(4)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count

            model.scale = torch.tensor(4.0)
            torch.testing.assert_close(compiled(x), model(x))

            model.scale.resize_(4).fill_(6.0)
            torch.testing.assert_close(compiled(x), model(x))

            model.offsets[0] = 5.0
            torch.testing.assert_close(compiled(x), model(x))

            global_bias = torch.tensor(7.0)
            torch.testing.assert_close(compiled(x), model(x))

            model.scale = torch.tensor(6.0)
            x = torch.ones(9)
            torch.testing.assert_close(compiled(x), model(x))

            def alias_sensitive(a, b):
                return a + b if a is b else a - b

            compiled_alias = torch.compile(
                alias_sensitive, backend="eager", fullgraph=True
            )
            a = torch.ones(4)
            b = torch.full((4,), 2.0)
            torch.testing.assert_close(compiled_alias(a, a), alias_sensitive(a, a))
            torch.testing.assert_close(compiled_alias(a, b), alias_sensitive(a, b))
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_plan_is_per_cache_entry(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mode = 1

                def forward(self, x):
                    return x + self.mode + GLOBAL_DICT["used"]

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=False
            )
            inputs = (torch.zeros(2), torch.zeros(3))
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                for x in inputs:
                    torch.testing.assert_close(
                        compiled(x), torch.full_like(x, 2.0)
                    )
            assert counter.frame_count == 2, counter.frame_count

            model.mode = 5
            for i, x in enumerate(inputs):
                GLOBAL_DICT["noise"] = [100 + i]
                torch.testing.assert_close(
                    compiled(x), torch.full_like(x, 6.0)
                )
            assert counter.frame_count == 4, counter.frame_count
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_preserves_cross_slice_alias_relations(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("value", torch.ones(2))

                def forward(self, x):
                    return self.value + x + GLOBAL_DICT["used"]

            aliased_model = Model()
            aliased_counter = CompileCounter()
            aliased = torch.compile(
                aliased_model, backend=aliased_counter, fullgraph=True
            )
            same = aliased_model.value
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                torch.testing.assert_close(aliased(same), torch.full((2,), 3.0))
            assert aliased_counter.frame_count == 1, aliased_counter.frame_count

            GLOBAL_DICT["noise"] = [100]
            different = torch.zeros_like(same)
            torch.testing.assert_close(aliased(different), torch.full((2,), 2.0))
            assert aliased_counter.frame_count == 2, aliased_counter.frame_count

            distinct_model = Model()
            distinct_counter = CompileCounter()
            distinct = torch.compile(
                distinct_model, backend=distinct_counter, fullgraph=True
            )
            for i in range(8):
                GLOBAL_DICT["noise"] = [i + 200]
                current = torch.full((2,), float(i + 2))
                torch.testing.assert_close(
                    distinct(current), distinct_model.value + current + 1
                )
            assert distinct_counter.frame_count == 1, distinct_counter.frame_count

            GLOBAL_DICT["noise"] = [300]
            torch.testing.assert_close(
                distinct(distinct_model.value), torch.full((2,), 3.0)
            )
            assert distinct_counter.frame_count == 2, distinct_counter.frame_count
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_preserves_tensor_no_hasattr_guard(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._cached_tensor = torch.ones(2)

                def forward(self, x):
                    return self._cached_tensor + x + GLOBAL_DICT["used"]

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )
            x = torch.zeros(2)

            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            assert counter.frame_count == 1, counter.frame_count

            model._cached_tensor._dynamo_dynamic_indices = set()
            GLOBAL_DICT["noise"] = [100]
            torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            assert counter.frame_count == 2, counter.frame_count
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_refreshes_unrelated_tensor_type_change(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._cached_tensor = torch.ones(2)

                def forward(self, x):
                    return self._cached_tensor + x

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count

            torch.Tensor._fastguard_unrelated_type_change = None
            try:
                torch.testing.assert_close(compiled(x), torch.ones(2))
                assert counter.frame_count == 1, counter.frame_count
            finally:
                del torch.Tensor._fastguard_unrelated_type_change
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_type_proof_fails_closed_on_class_attr(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._cached_tensor = torch.ones(2)

                def forward(self, x):
                    return self._cached_tensor + x

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count

            torch.Tensor._dynamo_dynamic_indices = set()
            try:
                torch.testing.assert_close(compiled(x), torch.ones(2))
                assert counter.frame_count == 2, counter.frame_count
            finally:
                del torch.Tensor._dynamo_dynamic_indices
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_rejects_ordinary_no_hasattr(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Model(torch.nn.Module):
                def forward(self, x):
                    if hasattr(self, "scale"):
                        return x + self.scale + GLOBAL_DICT["used"]
                    return x + 1 + GLOBAL_DICT["used"]

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            assert counter.frame_count == 1, counter.frame_count

            model.scale = torch.full((2,), 5.0)
            GLOBAL_DICT["noise"] = [100]
            torch.testing.assert_close(compiled(x), torch.full((2,), 6.0))
            assert counter.frame_count == 2, counter.frame_count
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_capability_census_is_training_only_diagnostic(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._cached_tensor = torch.ones(2)
                    self.offsets = [1.0]
                    self.device = torch.device("cpu")

                def helper(self, x):
                    return x

                def forward(self, x):
                    return self.helper(
                        self._cached_tensor
                        + x.to(self.device)
                        + self.offsets[0]
                    )

            guards = torch._C._dynamo.guards
            guards._reset_guard_fast_plan_capability_census()
            counter = CompileCounter()
            compiled = torch.compile(Model(), backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            assert counter.frame_count == 1, counter.frame_count

            census = guards._get_guard_fast_plan_capability_census()
            assert census["enabled"], census
            assert census["leaf_observations"] > 0, census
            assert census["accessor_observations"] > 0, census
            assert census["get_attr_accessors"] > 0, census
            assert census["owner_path_records"] > 0, census
            assert 0 < census["owner_path_unique_records"] <= census[
                "owner_path_records"
            ], census
            assert (
                census["generic_dict_binding_records"]
                + census["generic_dict_binding_unsupported"]
                + census["instance_attr_binding_records"]
                + census["type_method_binding_records"]
                + census["instance_attr_binding_unsupported"]
                == census["owner_path_records"]
            ), census
            assert census["generic_dict_binding_records"] > 0, census
            assert census["generic_dict_binding_unsupported"] == 0, census
            assert census["generic_dict_owner_proofs"] > 0, census
            assert census["generic_dict_exact_dict_tokens_removed"] > 0, census
            assert census["instance_attr_binding_records"] > 0, census
            assert census["type_method_binding_records"] > 0, census
            assert census["instance_attr_unique_owners"] > 0, census
            assert census["instance_attr_unique_types"] > 0, census
            classified_accessors = sum(
                census[name]
                for name in (
                    "get_attr_accessors",
                    "generic_get_attr_accessors",
                    "get_generic_dict_accessors",
                    "frame_locals_accessors",
                    "dict_getitem_accessors",
                    "list_getitem_accessors",
                    "tuple_getitem_accessors",
                    "get_item_accessors",
                    "set_getitem_accessors",
                    "tensor_property_accessors",
                    "indexed_accessors",
                    "grad_accessors",
                    "func_defaults_accessors",
                    "func_kw_defaults_accessors",
                    "globals_accessors",
                    "type_accessors",
                    "type_dict_accessors",
                    "type_mro_accessors",
                    "tuple_iterator_getitem_accessors",
                    "global_weakref_accessors",
                    "weakref_call_accessors",
                    "code_accessors",
                    "closure_accessors",
                    "call_function_no_args_accessors",
                    "python_lambda_accessors",
                    "other_accessors",
                )
            )
            assert classified_accessors == census["accessor_observations"], census
            assert (
                census["code_accessor_function_observations"]
                + census["code_accessor_bound_method_observations"]
                + census["code_accessor_instance_method_observations"]
                + census["code_accessor_unsupported_observations"]
                == census["code_accessors"]
            ), census
            assert (
                census["type_accessor_unique_owners"]
                <= census["type_accessors"]
            ), census
            assert (
                census["type_accessor_unique_types"]
                <= census["type_accessor_unique_owners"]
            ), census
            assert (
                census["code_accessor_unique_functions"]
                <= census["code_accessors"]
            ), census
            assert (
                census["code_accessor_unique_codes"]
                <= census["code_accessors"]
            ), census
            assert (
                census["type_accessor_any_proof_covered_owners"]
                + census["type_accessor_uncovered_owners"]
                == census["type_accessor_unique_owners"]
            ), census
            assert census["type_accessor_coverage_failures"] == 0, census
            assert census["code_accessor_proofs"] > 0, census
            assert census["unsupported_leaf_capabilities"] == 0, census
            assert census["unsupported_accessor_capabilities"] == 0, census
            assert census["equals_safe_constant_admissions"] > 0, census
            assert (
                sum(census["unsupported_leaf_capability_reasons"].values())
                == census["unsupported_leaf_capabilities"]
            ), census
            assert (
                census["last_unsupported_leaf_capability_reason"] == "none"
            ), census
            for name in (
                "type_accessor_generic_dict_covered_owners",
                "type_accessor_instance_attr_covered_owners",
                "type_accessor_type_method_covered_owners",
            ):
                assert (
                    census[name]
                    <= census["type_accessor_any_proof_covered_owners"]
                ), census

            guards._reset_guard_fast_plan_capability_census()
            census = guards._get_guard_fast_plan_capability_census()
            assert census["leaf_observations"] == 0, census
            assert census["accessor_observations"] == 0, census
            assert census["owner_path_records"] == 0, census
            assert census["owner_path_unique_records"] == 0, census
            assert census["generic_dict_binding_records"] == 0, census
            assert census["generic_dict_binding_unsupported"] == 0, census
            assert census["generic_dict_unique_owners"] == 0, census
            assert census["generic_dict_owner_proofs"] == 0, census
            assert census["generic_dict_owner_misses"] == 0, census
            assert census["generic_dict_exact_dict_tokens_removed"] == 0, census
            assert census["generic_dict_accessor_owner_proofs_removed"] == 0, census
            assert census["instance_attr_binding_records"] == 0, census
            assert census["type_method_binding_records"] == 0, census
            assert census["instance_attr_binding_unsupported"] == 0, census
            assert census["instance_attr_non_default_getattribute"] == 0, census
            assert census["instance_attr_non_default_heap_type"] == 0, census
            assert census["instance_attr_non_default_static_type"] == 0, census
            assert census["instance_attr_non_default_exact_tensor"] == 0, census
            assert census["instance_attr_non_default_exact_module"] == 0, census
            assert (
                census["instance_attr_non_default_exact_type_object"] == 0
            ), census
            assert (
                census["instance_attr_non_default_exact_dict_value"] == 0
            ), census
            assert (
                census["instance_attr_non_default_type_attr_absent"] == 0
            ), census
            assert census["instance_attr_default_unsupported"] == 0, census
            assert (
                census["instance_attr_default_unsupported_non_unicode_key"] == 0
            ), census
            assert (
                census["instance_attr_default_unsupported_type_version"] == 0
            ), census
            assert (
                census[
                    "instance_attr_default_unsupported_instance_dict_shadow"
                ]
                == 0
            ), census
            assert (
                census["instance_attr_default_shadow_plain_value"] == 0
            ), census
            assert (
                census[
                    "instance_attr_default_shadow_static_non_data_descriptor"
                ]
                == 0
            ), census
            assert (
                census[
                    "instance_attr_default_shadow_heap_non_data_descriptor"
                ]
                == 0
            ), census
            assert (
                census["instance_attr_default_shadow_unique_owner_keys"] == 0
            ), census
            assert (
                census["instance_attr_default_shadow_unique_type_keys"] == 0
            ), census
            assert (
                census["instance_attr_default_unsupported_data_descriptor"] == 0
            ), census
            assert (
                census[
                    "instance_attr_default_data_descriptor_unique_owner_keys"
                ]
                == 0
            ), census
            assert (
                census[
                    "instance_attr_default_unsupported_non_data_descriptor"
                ]
                == 0
            ), census
            assert (
                census["instance_attr_default_unsupported_type_attr_absent"] == 0
            ), census
            assert (
                census["instance_attr_default_unsupported_other"] == 0
            ), census
            assert (
                census["instance_attr_non_default_unique_types"] == 0
            ), census
            assert (
                census["instance_attr_non_default_unique_slots"] == 0
            ), census
            assert census["instance_attr_unique_owners"] == 0, census
            assert census["instance_attr_unique_types"] == 0, census
            assert census["type_method_owner_proofs"] == 0, census
            assert census["type_method_type_proofs"] == 0, census
            assert census["type_method_owner_misses"] == 0, census
            assert census["type_method_type_misses"] == 0, census
            assert census["type_method_type_refreshes"] == 0, census
            assert census["instance_attr_owner_proofs"] == 0, census
            assert census["instance_attr_type_proofs"] == 0, census
            assert census["instance_attr_owner_misses"] == 0, census
            assert census["instance_attr_type_misses"] == 0, census
            assert census["instance_attr_type_refreshes"] == 0, census
            assert census["instance_attr_shadow_binding_records"] == 0, census
            assert census["instance_attr_shadow_type_keys"] == 0, census
            assert census["instance_attr_dynamic_binding_records"] == 0, census
            assert census["instance_attr_dynamic_proofs"] == 0, census
            assert census["instance_attr_dynamic_misses"] == 0, census
            assert census["static_module_attr_binding_records"] == 0, census
            assert census["static_module_attr_owner_proofs"] == 0, census
            assert census["static_module_attr_type_proofs"] == 0, census
            assert (
                census["static_module_attr_owner_proofs_removed"] == 0
            ), census
            assert census["static_module_attr_owner_misses"] == 0, census
            assert census["static_module_attr_type_misses"] == 0, census
            assert (
                census["static_module_dynamic_attr_binding_records"] == 0
            ), census
            assert census["static_module_dynamic_attr_proofs"] == 0, census
            assert census["static_module_dynamic_attr_misses"] == 0, census
            assert census["static_type_attr_binding_records"] == 0, census
            assert census["static_type_attr_owner_proofs"] == 0, census
            assert census["static_type_attr_type_proofs"] == 0, census
            assert (
                census["static_type_attr_owner_proofs_removed"] == 0
            ), census
            assert census["static_type_attr_owner_misses"] == 0, census
            assert census["static_type_attr_type_misses"] == 0, census
            assert (
                census["static_type_dynamic_attr_binding_records"] == 0
            ), census
            assert census["static_type_dynamic_attr_proofs"] == 0, census
            assert census["static_type_dynamic_attr_misses"] == 0, census
            assert census["type_accessor_unique_owners"] == 0, census
            assert census["type_accessor_unique_types"] == 0, census
            assert census["code_accessor_unique_functions"] == 0, census
            assert census["code_accessor_unique_codes"] == 0, census
            assert (
                census["type_accessor_any_proof_covered_owners"] == 0
            ), census
            assert census["type_accessor_uncovered_owners"] == 0, census
            assert census["type_accessor_coverage_failures"] == 0, census
            assert census["code_accessor_proofs"] == 0, census
            assert census["code_accessor_misses"] == 0, census
            assert census["unsupported_leaf_capabilities"] == 0, census
            assert census["unsupported_accessor_capabilities"] == 0, census
            assert census["equals_safe_constant_admissions"] == 0, census
            assert census["equals_exact_set_token_emissions"] == 0, census
            assert census["equals_exact_set_token_items"] == 0, census
            assert census["equals_exact_set_token_max_size"] == 0, census
            assert census["equals_exact_set_token_misses"] == 0, census
            assert census["unsupported_equals_types"] == {}, census
            assert (
                sum(census["unsupported_leaf_capability_reasons"].values())
                == 0
            ), census
            assert (
                census["last_unsupported_leaf_capability_reason"] == "none"
            ), census

            class CustomGetattributeModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.ones(2)

                def __getattribute__(self, name):
                    return object.__getattribute__(self, name)

                def forward(self, x):
                    return self.scale + x

            custom_counter = CompileCounter()
            custom_compiled = torch.compile(
                CustomGetattributeModel(),
                backend=custom_counter,
                fullgraph=True,
            )
            for _ in range(8):
                torch.testing.assert_close(
                    custom_compiled(x), torch.ones(2)
                )
            assert custom_counter.frame_count == 1, custom_counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert census["instance_attr_non_default_getattribute"] > 0, census
            assert (
                census["instance_attr_non_default_heap_type"]
                + census["instance_attr_non_default_static_type"]
                == census["instance_attr_non_default_getattribute"]
            ), census
            assert census["instance_attr_non_default_heap_type"] > 0, census
            assert (
                census["instance_attr_non_default_exact_dict_value"] > 0
            ), census
            assert (
                census["instance_attr_non_default_type_attr_absent"] > 0
            ), census
            assert (
                census["instance_attr_non_default_unique_types"] > 0
            ), census
            assert (
                census["instance_attr_non_default_unique_slots"] > 0
            ), census
            assert (
                census["instance_attr_binding_unsupported"]
                >= census["instance_attr_non_default_getattribute"]
            ), census
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_retains_compiled_self_lifetime(self):
        script = """
            import gc
            import weakref

            import torch
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.ones(2)

                def forward(self, x):
                    return self.scale + x

            model = Model()
            model_ref = weakref.ref(model)
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count

            del model
            gc.collect()
            assert model_ref() is not None
            torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_static_module_attr_binding_proof(self):
        script = """
            import types

            import torch
            from torch._dynamo.testing import CompileCounter

            guards = torch._C._dynamo.guards
            namespace = types.ModuleType("fastguard_test_namespace")
            namespace.scale = torch.ones(2)

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = namespace

                def forward(self, x):
                    return self.namespace.scale + x

            guards._reset_guard_fast_plan_capability_census()
            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count

            census = guards._get_guard_fast_plan_capability_census()
            assert census["static_module_attr_binding_records"] > 0, census
            assert census["static_module_attr_type_proofs"] > 0, census
            assert census["instance_attr_non_default_exact_module"] > 0, census
            assert census["static_module_attr_owner_misses"] == 0, census
            assert census["static_module_attr_type_misses"] == 0, census

            namespace.scale = torch.full((2,), 3.0)
            torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
            assert counter.frame_count == 2, counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert (
                census["static_module_attr_owner_misses"]
                + census["generic_dict_owner_misses"]
                > 0
            ), census

            dynamic_values = [torch.ones(2)]

            def module_getattr(name):
                if name == "dynamic_scale":
                    return dynamic_values[0]
                raise AttributeError(name)

            namespace.__getattr__ = module_getattr

            class ModuleGetattrModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = namespace

                def forward(self, x):
                    return self.namespace.dynamic_scale + x

            guards._reset_guard_fast_plan_capability_census()
            module_getattr_counter = CompileCounter()
            module_getattr_compiled = torch.compile(
                ModuleGetattrModel(),
                backend=module_getattr_counter,
                fullgraph=True,
            )
            for _ in range(8):
                torch.testing.assert_close(
                    module_getattr_compiled(x), torch.ones(2)
                )
            census = guards._get_guard_fast_plan_capability_census()
            assert (
                census["static_module_dynamic_attr_binding_records"] > 0
            ), census
            assert census["static_module_dynamic_attr_proofs"] > 0, census
            assert census["static_module_dynamic_attr_misses"] == 0, census

            dynamic_values[0] = torch.full((2,), 4.0)
            torch.testing.assert_close(
                module_getattr_compiled(x), torch.full((2,), 4.0)
            )
            assert module_getattr_counter.frame_count == 2, (
                module_getattr_counter.frame_count
            )
            census = guards._get_guard_fast_plan_capability_census()
            assert census["static_module_dynamic_attr_misses"] > 0, census

            class DynamicModule(types.ModuleType):
                def __getattribute__(self, name):
                    return super().__getattribute__(name)

            dynamic = DynamicModule("fastguard_dynamic_namespace")
            dynamic.scale = torch.ones(2)

            class DynamicModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = dynamic

                def forward(self, x):
                    return self.namespace.scale + x

            guards._reset_guard_fast_plan_capability_census()
            dynamic_counter = CompileCounter()
            dynamic_compiled = torch.compile(
                DynamicModel(), backend=dynamic_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(dynamic_compiled(x), torch.ones(2))
            census = guards._get_guard_fast_plan_capability_census()
            assert census["static_module_attr_binding_records"] == 0, census
            assert census["instance_attr_binding_unsupported"] > 0, census
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_static_type_attr_binding_proof(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            guards = torch._C._dynamo.guards

            class Namespace:
                scale = torch.ones(2)

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = Namespace

                def forward(self, x):
                    return self.namespace.scale + x

            guards._reset_guard_fast_plan_capability_census()
            counter = CompileCounter()
            compiled = torch.compile(Model(), backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert census["static_type_attr_binding_records"] > 0, census
            assert census["static_type_attr_type_proofs"] > 0, census
            assert census["static_type_attr_owner_misses"] == 0, census
            assert census["static_type_attr_type_misses"] == 0, census

            Namespace.scale = torch.full((2,), 3.0)
            torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
            assert counter.frame_count == 2, counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert (
                census["static_type_attr_owner_misses"]
                + census["generic_dict_owner_misses"]
                > 0
            ), census

            class Descriptor:
                def __init__(self):
                    self.value = torch.ones(2)

                def __get__(self, obj, owner):
                    return self.value

            descriptor = Descriptor()

            class DynamicNamespace:
                scale = descriptor

            class DynamicModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = DynamicNamespace

                def forward(self, x):
                    return self.namespace.scale + x

            guards._reset_guard_fast_plan_capability_census()
            dynamic_counter = CompileCounter()
            dynamic_compiled = torch.compile(
                DynamicModel(), backend=dynamic_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(dynamic_compiled(x), torch.ones(2))
            census = guards._get_guard_fast_plan_capability_census()
            assert (
                census["static_type_dynamic_attr_binding_records"] > 0
            ), census
            assert census["static_type_dynamic_attr_proofs"] > 0, census
            assert census["static_type_dynamic_attr_misses"] == 0, census

            descriptor.value = torch.full((2,), 4.0)
            torch.testing.assert_close(
                dynamic_compiled(x), torch.full((2,), 4.0)
            )
            assert dynamic_counter.frame_count == 2, dynamic_counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert census["static_type_dynamic_attr_misses"] > 0, census

            class CustomMeta(type):
                pass

            class CustomNamespace(metaclass=CustomMeta):
                scale = torch.ones(2)

            class CustomModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = CustomNamespace

                def forward(self, x):
                    return self.namespace.scale + x

            guards._reset_guard_fast_plan_capability_census()
            custom_counter = CompileCounter()
            custom_compiled = torch.compile(
                CustomModel(), backend=custom_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(custom_compiled(x), torch.ones(2))
            census = guards._get_guard_fast_plan_capability_census()
            assert census["static_type_attr_binding_records"] == 0, census
            assert census["instance_attr_binding_unsupported"] > 0, census
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_exact_set_equals_token_detects_mutation(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mode = {1, 2}

                def forward(self, x):
                    if self.mode == {1, 2}:
                        return x + 1
                    return x - 1

            guards = torch._C._dynamo.guards
            guards._reset_guard_fast_plan_capability_census()
            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count

            census = guards._get_guard_fast_plan_capability_census()
            assert census["equals_exact_set_token_emissions"] > 0, census
            assert census["unsupported_leaf_capabilities"] == 0, census

            model.mode.add(3)
            torch.testing.assert_close(compiled(x), torch.full((2,), -1.0))
            assert counter.frame_count == 2, counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert census["equals_exact_set_token_misses"] > 0, census
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_code_accessor_proof_detects_code_mutation(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def helper(self, x):
                    return x + 1

                def forward(self, x):
                    return self.helper(x)

            guards = torch._C._dynamo.guards
            guards._reset_guard_fast_plan_capability_census()
            counter = CompileCounter()
            model = Model()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count
            stats = guards._get_guard_fast_plan_capability_census()
            assert stats["code_accessor_proofs"] > 0, stats
            assert stats["type_accessor_coverage_failures"] == 0, stats

            original_code = Model.helper.__code__
            try:
                def replacement(self, x):
                    return x + 2

                Model.helper.__code__ = replacement.__code__
                torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
                assert counter.frame_count == 2, counter.frame_count
                stats = guards._get_guard_fast_plan_capability_census()
                assert stats["code_accessor_misses"] > 0, stats
            finally:
                Model.helper.__code__ = original_code
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_type_method_binding_proof(self):
        script = """
            import types

            import torch
            from torch._dynamo.testing import CompileCounter

            guards = torch._C._dynamo.guards
            GLOBAL_DICT = {"used": 1, "noise": [0]}

            def warm(compiled, x, expected):
                for i in range(8):
                    GLOBAL_DICT["noise"] = [i]
                    torch.testing.assert_close(compiled(x), expected)

            class InstanceShadowModel(torch.nn.Module):
                def helper(self, x):
                    return x + 1

                def forward(self, x):
                    return self.helper(x) + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            model = InstanceShadowModel()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            warm(compiled, x, torch.full((2,), 2.0))
            census = guards._get_guard_fast_plan_capability_census()
            assert census["type_method_owner_proofs"] > 0, census
            assert census["type_method_type_proofs"] > 0, census

            def replacement(self, value):
                return value + 4

            model.helper = types.MethodType(replacement, model)
            GLOBAL_DICT["noise"] = [100]
            torch.testing.assert_close(compiled(x), torch.full((2,), 5.0))
            assert counter.frame_count == 2, counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert (
                census["type_method_owner_misses"]
                + census["generic_dict_owner_misses"]
                > 0
            ), census

            class ClassMutationModel(torch.nn.Module):
                def helper(self, x):
                    return x + 1

                def forward(self, x):
                    return self.helper(x) + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            class_model = ClassMutationModel()
            class_counter = CompileCounter()
            class_compiled = torch.compile(
                class_model, backend=class_counter, fullgraph=True
            )
            warm(class_compiled, x, torch.full((2,), 2.0))
            original = ClassMutationModel.helper
            ClassMutationModel.helper = replacement
            try:
                GLOBAL_DICT["noise"] = [200]
                torch.testing.assert_close(
                    class_compiled(x), torch.full((2,), 5.0)
                )
                assert class_counter.frame_count == 2, class_counter.frame_count
                census = guards._get_guard_fast_plan_capability_census()
                assert census["type_method_type_misses"] > 0, census
            finally:
                ClassMutationModel.helper = original

            class RefreshModel(torch.nn.Module):
                def helper(self, x):
                    return x + 1

                def forward(self, x):
                    return self.helper(x) + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            refresh_model = RefreshModel()
            refresh_counter = CompileCounter()
            refresh_compiled = torch.compile(
                refresh_model, backend=refresh_counter, fullgraph=True
            )
            warm(refresh_compiled, x, torch.full((2,), 2.0))
            RefreshModel._fastguard_unrelated_type_change = None
            try:
                GLOBAL_DICT["noise"] = [300]
                torch.testing.assert_close(
                    refresh_compiled(x), torch.full((2,), 2.0)
                )
                assert refresh_counter.frame_count == 1, refresh_counter.frame_count
                census = guards._get_guard_fast_plan_capability_census()
                assert census["type_method_type_refreshes"] > 0, census
            finally:
                del RefreshModel._fastguard_unrelated_type_change
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_instance_attr_binding_proof(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            guards = torch._C._dynamo.guards
            GLOBAL_DICT = {"used": 1, "noise": [0]}

            def warm(compiled, x, expected):
                for i in range(8):
                    GLOBAL_DICT["noise"] = [i]
                    torch.testing.assert_close(compiled(x), expected)

            class InstanceMutationModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.ones(2)

                def forward(self, x):
                    return self.scale + x + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            model = InstanceMutationModel()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            warm(compiled, x, torch.full((2,), 2.0))
            census = guards._get_guard_fast_plan_capability_census()
            assert census["instance_attr_owner_proofs"] > 0, census
            assert census["instance_attr_type_proofs"] > 0, census

            model.scale = torch.full((2,), 3.0)
            GLOBAL_DICT["noise"] = [100]
            torch.testing.assert_close(compiled(x), torch.full((2,), 4.0))
            assert counter.frame_count == 2, counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert (
                census["instance_attr_owner_misses"]
                + census["generic_dict_owner_misses"]
                > 0
            ), census

            class DescriptorMutationModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.ones(2)

                def forward(self, x):
                    return self.scale + x + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            descriptor_model = DescriptorMutationModel()
            descriptor_counter = CompileCounter()
            descriptor_compiled = torch.compile(
                descriptor_model, backend=descriptor_counter, fullgraph=True
            )
            warm(descriptor_compiled, x, torch.full((2,), 2.0))
            DescriptorMutationModel.scale = property(
                lambda self: torch.full((2,), 5.0)
            )
            try:
                GLOBAL_DICT["noise"] = [200]
                torch.testing.assert_close(
                    descriptor_compiled(x), torch.full((2,), 6.0)
                )
                assert descriptor_counter.frame_count == 2, (
                    descriptor_counter.frame_count
                )
                census = guards._get_guard_fast_plan_capability_census()
                assert census["instance_attr_type_misses"] > 0, census
            finally:
                del DescriptorMutationModel.scale

            class ShadowValueModel(torch.nn.Module):
                scale = None

                def __init__(self):
                    super().__init__()
                    self.scale = torch.ones(2)

                def forward(self, x):
                    return self.scale + x + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            shadow_model = ShadowValueModel()
            shadow_counter = CompileCounter()
            shadow_compiled = torch.compile(
                shadow_model, backend=shadow_counter, fullgraph=True
            )
            warm(shadow_compiled, x, torch.full((2,), 2.0))
            census = guards._get_guard_fast_plan_capability_census()
            assert census["instance_attr_shadow_binding_records"] > 0, census
            assert census["instance_attr_shadow_type_keys"] > 0, census

            ShadowValueModel.scale = property(
                lambda self: torch.full((2,), 5.0)
            )
            try:
                GLOBAL_DICT["noise"] = [250]
                torch.testing.assert_close(
                    shadow_compiled(x), torch.full((2,), 6.0)
                )
                assert shadow_counter.frame_count == 2, shadow_counter.frame_count
                census = guards._get_guard_fast_plan_capability_census()
                assert census["instance_attr_type_misses"] > 0, census
            finally:
                ShadowValueModel.scale = None

            class InitialScaleDescriptor:
                def __get__(self, obj, owner):
                    if obj is None:
                        return self
                    return obj._scale

                def __set__(self, obj, value):
                    obj._scale = value

            initial_scale_descriptor = InitialScaleDescriptor()

            class DynamicDescriptorModel(torch.nn.Module):
                scale = initial_scale_descriptor

                def __init__(self):
                    super().__init__()
                    self._scale = torch.ones(2)

                def forward(self, x):
                    return self.scale + x + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            dynamic_model = DynamicDescriptorModel()
            dynamic_counter = CompileCounter()
            dynamic_compiled = torch.compile(
                dynamic_model, backend=dynamic_counter, fullgraph=True
            )
            warm(dynamic_compiled, x, torch.full((2,), 2.0))
            census = guards._get_guard_fast_plan_capability_census()
            assert census["instance_attr_dynamic_binding_records"] > 0, census
            assert census["instance_attr_dynamic_proofs"] > 0, census

            DynamicDescriptorModel.scale = property(
                lambda self: torch.full((2,), 5.0)
            )
            try:
                GLOBAL_DICT["noise"] = [275]
                torch.testing.assert_close(
                    dynamic_compiled(x), torch.full((2,), 6.0)
                )
                assert dynamic_counter.frame_count == 2, dynamic_counter.frame_count
                census = guards._get_guard_fast_plan_capability_census()
                assert census["instance_attr_dynamic_misses"] > 0, census
            finally:
                DynamicDescriptorModel.scale = initial_scale_descriptor

            class RefreshModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.ones(2)

                def forward(self, x):
                    return self.scale + x + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            refresh_model = RefreshModel()
            refresh_counter = CompileCounter()
            refresh_compiled = torch.compile(
                refresh_model, backend=refresh_counter, fullgraph=True
            )
            warm(refresh_compiled, x, torch.full((2,), 2.0))
            RefreshModel._fastguard_unrelated_type_change = None
            try:
                GLOBAL_DICT["noise"] = [300]
                torch.testing.assert_close(
                    refresh_compiled(x), torch.full((2,), 2.0)
                )
                assert refresh_counter.frame_count == 1, refresh_counter.frame_count
                census = guards._get_guard_fast_plan_capability_census()
                assert census["instance_attr_type_refreshes"] > 0, census
            finally:
                del RefreshModel._fastguard_unrelated_type_change
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_clears_dynamic_descriptor_exception(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            guards = torch._C._dynamo.guards
            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class FlakyDescriptor:
                def __init__(self):
                    self.fail_next = False

                def __get__(self, obj, owner):
                    if obj is None:
                        return self
                    if self.fail_next:
                        self.fail_next = False
                        raise RuntimeError("transient descriptor failure")
                    return obj._scale

                def __set__(self, obj, value):
                    obj._scale = value

            descriptor = FlakyDescriptor()

            class Model(torch.nn.Module):
                scale = descriptor

                def __init__(self):
                    super().__init__()
                    self._scale = torch.ones(2)

                def forward(self, x):
                    return self.scale + x + GLOBAL_DICT["used"]

            guards._reset_guard_fast_plan_capability_census()
            counter = CompileCounter()
            compiled = torch.compile(Model(), backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            assert counter.frame_count == 1, counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert census["instance_attr_dynamic_proofs"] > 0, census

            descriptor.fail_next = True
            GLOBAL_DICT["noise"] = [100]
            torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            census = guards._get_guard_fast_plan_capability_census()
            assert census["instance_attr_dynamic_misses"] > 0, census

            GLOBAL_DICT["noise"] = [101]
            torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_generic_dict_binding_proof(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            guards = torch._C._dynamo.guards
            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Holder:
                def __init__(self):
                    self.scale = torch.ones(2)

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.holder = Holder()

                def forward(self, x):
                    return (
                        self.holder.__dict__["scale"]
                        + x
                        + GLOBAL_DICT["used"]
                    )

            guards._reset_guard_fast_plan_capability_census()
            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            assert counter.frame_count == 1, counter.frame_count
            census = guards._get_guard_fast_plan_capability_census()
            assert census["generic_dict_owner_proofs"] > 0, census
            assert census["generic_dict_exact_dict_tokens_removed"] > 0, census

            model.holder.__dict__ = dict(model.holder.__dict__)
            GLOBAL_DICT["noise"] = [100]
            torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            census = guards._get_guard_fast_plan_capability_census()
            assert census["generic_dict_owner_misses"] > 0, census
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        env["TORCHDYNAMO_GUARD_FAST_PLAN_CAPABILITY_CENSUS"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

    def test_actual_partial_misses_on_data_descriptor_install(self):
        script = """
            import torch
            from torch._dynamo.testing import CompileCounter

            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.ones(2)

                def forward(self, x):
                    return self.scale + x + GLOBAL_DICT["used"]

            class ScaleDescriptor:
                def __get__(self, obj, owner):
                    return torch.full((2,), 5.0)

                def __set__(self, obj, value):
                    obj.__dict__["scale"] = value

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                torch.testing.assert_close(compiled(x), torch.full((2,), 2.0))
            assert counter.frame_count == 1, counter.frame_count

            Model.scale = ScaleDescriptor()
            try:
                GLOBAL_DICT["noise"] = [100]
                torch.testing.assert_close(compiled(x), torch.full((2,), 6.0))
                assert counter.frame_count == 2, counter.frame_count
            finally:
                del Model.scale
        """
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
