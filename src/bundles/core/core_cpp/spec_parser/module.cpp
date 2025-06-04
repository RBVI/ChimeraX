// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <cstdlib>
#include <vector>
#include <utility>

#include "Python.h"
#include "peglib.h"

//#include <logger/logger.h>

using namespace peg;

#if 0
typedef std::vector<unsigned char> RGBA;
#endif

static parser spec_parser;
#if 0
static PyObject* objects_class;
static PyObject* objects_intersect;
static PyObject* objects_union;
static PyObject* get_selector;
static PyObject* add_implied_bonds;
#endif
static PyObject* session;
#if 0
#ifdef EVALUATE
static PyObject* models;
#endif
static PyObject* combine_arg;
static PyObject* invert_arg;
static PyObject* list_arg;
static PyObject* add_model_arg;
static PyObject* add_atoms_arg;
static PyObject* add_parts_arg;
static PyObject* add_pseudobonds_arg;
static PyObject* atomspec_has_atoms_arg;
static PyObject* atomspec_has_pseudobonds_arg;
static PyObject* atomspec_model_attr_arg;
static PyObject* atomspec_pseudobonds_arg;
static PyObject* ColorArg;
static PyObject* make_converter;
static PyObject* converter;
static PyObject* uint8x4_arg;
static PyObject* attr_test_class;
static PyObject* op_eq;
static PyObject* op_ne;
static PyObject* op_ge;
static PyObject* op_gt;
static PyObject* op_le;
static PyObject* op_lt;
static PyObject* op_not;
static PyObject* op_truth;
static PyObject* _Chain_class;
static PyObject* _Residue_class;
static PyObject* _Atom_class;
static PyObject* _Part_class;
static PyObject* _PartList_class;
static PyObject* _add_model_parts;
#endif
static std::string use_python_error("Use Python error");
static bool add_implied, order_implicit_atoms, outermost_inversion;

static const char*
docstr_parse = \
"parse(session, text)\n" \
"\n" \
"Parse the given text for an initial atom spec and if one is found" \
" return an chimera.core.commands.atomspec.AtomSpec instance," \
" the part of text used for the atom spec, and the remaining text.";

#if 0
class Value {
public:
    // Can't put the next four fields in a union because the first two have constructors
    std::string  str;
    RGBA  rgba;
    long  integer;
    double  floating_point;
    enum ValType { STR_TYPE, RGBA_TYPE, INT_TYPE, FLOAT_TYPE };
    ValType  val_type;
    Value() { } // so that AttrTester can declare it
    Value(std::string v) { val_type = STR_TYPE; str = v; }
    Value(RGBA v) { val_type = RGBA_TYPE; rgba = v; }
    Value(long v) { val_type = INT_TYPE; integer = v; }
    Value(double v) { val_type = FLOAT_TYPE; floating_point = v; }
};

class AttrTester {
    std::string  name, op = "";
    bool  exists;
    Value  val;
    mutable PyObject*  _py_attr_test = nullptr;
    mutable bool  _incremented_ref_count = false;
public:
    AttrTester(std::string n, std::string o, Value v) { name = n; op = o; val = v; }
    AttrTester(std::string_view n, std::string o, Value v) { name = n; op = o; val = v; }
    AttrTester(std::string n, bool e) { name = n; exists = e; }
    AttrTester(std::string_view n, bool e) { name = n; exists = e; }
    virtual ~AttrTester() {
        if (_incremented_ref_count) { Py_DECREF(_py_attr_test); }
    }
    AttrTester(const AttrTester& obj) {
        name = obj.name;
        op = obj.op;
        exists = obj.exists;
        val = obj.val;
        _py_attr_test = obj._py_attr_test;
        if (_py_attr_test != nullptr) {
            Py_INCREF(_py_attr_test);
            _incremented_ref_count = true;
        } else
            _incremented_ref_count = false;
    }
    PyObject*  py_attr_test() const;
};

PyObject*  AttrTester::py_attr_test() const {
    if (_py_attr_test == nullptr) {
        PyObject *no_arg, *name_arg, *op_arg, *value_arg;
        op_arg = nullptr;
        if (op == "") {
            if (exists) {
                no_arg = Py_None;
                op_arg = op_truth;
            } else {
                no_arg = Py_True;
                op_arg = op_not;
            }
        } else
            no_arg = Py_None;

        name_arg = PyUnicode_FromString(name.c_str());

        if (op == "=")
            op_arg = op_eq;
        else if (op == "!=")
            op_arg = op_ne;
        else if (op == ">=")
            op_arg = op_ge;
        else if (op == ">")
            op_arg = op_gt;
        else if (op == "<=")
            op_arg = op_le;
        else if (op == "<")
            op_arg = op_lt;
        if (op_arg == nullptr)
            op_arg = PyUnicode_FromString(op.c_str());
        else
            Py_INCREF(op_arg);

        if (op == "")
            value_arg = Py_None;
        else {
            if (val.val_type == Value::STR_TYPE)
                value_arg = PyUnicode_FromString(val.str.c_str());
            else if (val.val_type == Value::RGBA_TYPE) {
                value_arg = PyTuple_New(val.rgba.size());
                if (value_arg == nullptr) {
                    Py_DECREF(name_arg);
                    Py_DECREF(op_arg);
                    throw std::runtime_error("Could not create tuple");
                }
                for (decltype(val.rgba)::size_type i = 0; i < val.rgba.size(); ++i) {
                    auto py_int = PyLong_FromLong(val.rgba[i]);
                    if (py_int == nullptr) {
                        Py_DECREF(name_arg);
                        Py_DECREF(op_arg);
                        Py_DECREF(value_arg);
                        throw std::runtime_error("Could not create Python integer");
                    }
                    PyTuple_SET_ITEM(value_arg, i, PyLong_FromLong(val.rgba[i]));
                }
            } else if (val.val_type == Value::INT_TYPE) {
                value_arg = PyLong_FromLong(val.integer);
                if (value_arg == nullptr) {
                    Py_DECREF(name_arg);
                    Py_DECREF(op_arg);
                    throw std::runtime_error("Could not create Python integer");
                }
            } else {
                value_arg = PyFloat_FromDouble(val.floating_point);
                if (value_arg == nullptr) {
                    Py_DECREF(name_arg);
                    Py_DECREF(op_arg);
                    throw std::runtime_error("Could not create Python float");
                }
            }
        }
        _py_attr_test = PyObject_CallFunctionObjArgs(attr_test_class,
            no_arg, name_arg, op_arg, value_arg, nullptr);
        if (_py_attr_test == nullptr) {
            Py_DECREF(name_arg);
            Py_DECREF(op_arg);
            Py_DECREF(value_arg);
            throw std::logic_error(use_python_error);
        }
    }
    _incremented_ref_count = true;
    return _py_attr_test;
}

class ModelPart {
public:
    bool  any = false;
    int   num;
    ModelPart() { any = true; }
    ModelPart(int x) { num = x; }
};

class ModelMatcher {
public:
    bool  range = false;
    ModelPart   start, end;
    ModelMatcher() { start.any = true; }
    ModelMatcher(ModelPart mp) { start = mp; }
    ModelMatcher(ModelPart mp1, ModelPart mp2) { range = true; start = mp1; end = mp2; }
    bool  matches_id_val(long id_val) {
        if (range)
            return (start.any || id_val >= start.num) && (end.any || id_val <= end.num);
        return start.any || id_val == start.num;
    }
};

class GlobalModelMatcher {
public:
    bool  exact_match;
    std::vector<std::vector<ModelMatcher>> levels;
    GlobalModelMatcher(bool em = false, std::vector<std::vector<ModelMatcher>> ls = {})
        { exact_match = em; levels = ls; }
    std::vector<PyObject*>  matches();
};

std::vector<PyObject*>
GlobalModelMatcher::matches()
{
    std::vector<PyObject*> matched_models;
    auto list_size = PySequence_Fast_GET_SIZE(models);
    for (decltype(list_size) i = 0; i < list_size; ++i) {
        auto m = PySequence_Fast_GET_ITEM(models, i);
        if (levels.empty()) {
            // matches all
            matched_models.push_back(m);
            continue;
        }
        auto id_tuple = PyObject_GetAttrString(m, "id");
        if (id_tuple == nullptr)
            throw std::logic_error(use_python_error);
        if (!PyTuple_Check(id_tuple)) {
            Py_DECREF(id_tuple);
            throw std::logic_error("Model ID is not a tuple");
        }
        auto num_levels = levels.size();
        decltype(num_levels) tuple_size = PyTuple_GET_SIZE(id_tuple);
        if ((exact_match && num_levels == tuple_size)
        || (!exact_match && num_levels <= tuple_size)) {
            bool matched_all_levels = true;
            for (decltype(num_levels) i = 0; i < num_levels; ++i) {
                auto& model_matchers = levels[i];
                auto id_item = PyTuple_GET_ITEM(id_tuple, i);
                if (!PyLong_Check(id_item)) {
                    Py_DECREF(id_tuple);
                    throw std::logic_error("Model ID tuple contains non-integer");
                }
                int id_val = PyLong_AsLong(id_item);
                bool matched_level = false;
                for (auto& model_matcher: model_matchers) {
                    if (model_matcher.matches_id_val(id_val)) {
                        matched_level = true;
                        break;
                    }
                }
                if (!matched_level) {
                    matched_all_levels = false;
                    break;
                }
            }
            if (matched_all_levels)
                matched_models.push_back(m);
        }
        Py_DECREF(id_tuple);
    }
    return matched_models;
}

static PyObject*
new_objects_instance()
{
    auto objects_inst = PyObject_CallNoArgs(objects_class);
    if (objects_inst == nullptr)
        throw std::runtime_error("Cannot create Objects instance");
    return objects_inst;
}
#endif

static size_t err_line, err_col;
static std::string err_msg;
static bool err_valid;

static void
set_error_info(PyObject* err_type, std::string msg)
{
    auto err_val = PyTuple_New(2);
    if (err_val == nullptr) {
        PyErr_SetString(PyExc_AssertionError, "Could not create error-value tuple");
        throw std::runtime_error("Could not create tuple");
    }
    PyTuple_SetItem(err_val, 0, PyLong_FromSize_t( err_valid ? err_col-1 : (size_t)0));
    if (PyErr_Occurred() == nullptr) {
        if (msg == use_python_error) {
            PyErr_SetString(PyExc_AssertionError, "Trying to use Python error when none set");
            throw std::runtime_error("No Python error message to use");
        }
        PyTuple_SetItem(err_val, 1, PyUnicode_FromString(msg.c_str()));
        PyErr_SetObject(err_type, err_val);
    } else {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        if (msg == use_python_error) {
            PyTuple_SetItem(err_val, 1, value);
        } else {
            PyTuple_SetItem(err_val, 1, PyUnicode_FromString(msg.c_str()));
            Py_DECREF(value);
        }
        PyErr_Restore(err_type, err_val, traceback);
        Py_DECREF(type);
    }
}

//TODO: accept find_match() keywords
extern "C" PyObject *
parse(PyObject *, PyObject *args)
{
    const char* text;
#ifdef DEBUG
    if (!PyArg_ParseTuple(args, "s", &text))
        return nullptr;
    std::shared_ptr<peg::Ast> ast;
    if (spec_parser.parse(text, ast)) {
        std::cout << "Raw AST: " << peg::ast_to_s(ast) << "\n";
        ast = spec_parser.optimize_ast(ast);
        std::cout << "Optimized AST: " << peg::ast_to_s(ast) << "\n";
    } else {
        std::cout << "parse failure for '" << text << "'\n";
    }
    return Py_None;
#else
#ifdef EVALUATE
    int c_quoted, c_add_implied, c_order_implicit;
    PyObject *parse_error_class, *semantics_error_class;
    if (!PyArg_ParseTuple(args, "OOspOOpp", &session, &models, &text, &c_quoted, &parse_error_class,
            &semantics_error_class, &c_add_implied, &c_order_implicit))
        return nullptr;
#endif
    int c_quoted, c_add_implied;
    PyObject *parse_error_class, *semantics_error_class;
    if (!PyArg_ParseTuple(args, "OspOOp", &session, &text, &c_quoted, &parse_error_class,
            &semantics_error_class, &c_add_implied))
        return nullptr;

    bool quoted = static_cast<bool>(c_quoted);
    add_implied = static_cast<bool>(c_add_implied);
    spec_parser.set_logger([](size_t line, size_t col, const std::string& msg) {
        err_valid = true;
        err_line = line;
        err_col = col;
        err_msg = msg;
    });
    //TODO: eval() is going to wind up being a big function (possibly calling subfunctions), so
    // change to static non-lambda
    std::function<PyObject*(const Ast&)> eval = [&](const Ast &ast) {
        std::cerr << ast.name << " '" << ast.token_to_string() << "'; choice " << ast.choice << "  " << ast.nodes.size() << " subnodes\n";
        for (auto node: ast.nodes) {
            eval(*node);
        }
        return Py_None;
    };
    std::shared_ptr<peg::Ast> ast;
    if (spec_parser.parse(text, ast)) {
        //TODO: Check if optimized AST is usable.  I suspect that ::name=="CYS" produces an unusable AST
        // because it skips levels
        eval(*ast);
        return Py_None;
    } else {
        if (quoted) {
            set_error_info(parse_error_class, err_msg);
            return nullptr;
        }
        //TODO: possibly try again
        set_error_info(parse_error_class, err_msg);
        return nullptr;
    }
#ifdef EVALUATE
    bool quoted = static_cast<bool>(c_quoted);
    add_implied = static_cast<bool>(c_add_implied);
    order_implicit_atoms = static_cast<bool>(c_order_implicit);
    models = PySequence_Fast(models, "parse() arg 'models' is not a sequence");
    if (models == nullptr)
        return nullptr;
    PyObject* returned_objects_instance = nullptr;
    std::string trial_text = text;
    spec_parser.set_logger([](size_t line, size_t col, const std::string& msg) {
        err_valid = true;
        err_line = line;
        err_col = col;
        err_msg = msg;
    });
    // logic_error for unexpected results from Python calls, e.g. add_implied_bonds throws an error
    // invalid_argument for parsing failure
    // runtime_error for basic failures like allocation failures or import failures
    outermost_inversion = false;
    try {
        err_valid = false;
        spec_parser.parse(text, returned_objects_instance);
        if (returned_objects_instance == nullptr && !quoted) {
            // progressively lop off spaced-separated text and see if what remain is legal...
            auto space_pos = trial_text.find_last_of(' ');
            while (space_pos != std::string::npos) {
                trial_text = trial_text.substr(0, space_pos);
                err_valid = false;
                spec_parser.parse(trial_text.c_str(), returned_objects_instance);
                if (returned_objects_instance != nullptr)
                    break;
                space_pos = trial_text.find_last_of(' ');
            }
        }
    } catch (std::runtime_error& e) {
        Py_DECREF(models);
        try {
            set_error_info(PyExc_RuntimeError, e.what());
        } catch (std::runtime_error &e) {
            return nullptr;
        }
        return nullptr;
    } catch (std::invalid_argument& e) {
        Py_DECREF(models);
        try {
            set_error_info(parse_error_class, e.what());
        } catch (std::runtime_error &e) {
            return nullptr;
        }
        return nullptr;
    } catch (std::logic_error& e) {
        // invalid_argument is a subclass of logic_error, so catch logic_error second
        Py_DECREF(models);
        try {
            set_error_info(semantics_error_class, e.what());
        } catch (std::runtime_error &e) {
            return nullptr;
        }
        return nullptr;
    }
    Py_DECREF(models);
    if (returned_objects_instance == nullptr) {
        set_error_info(parse_error_class, "parser did not set Objects instance");
        return nullptr;
    }
    if (add_implied) {
        auto ret = PyObject_CallOneArg(add_implied_bonds, returned_objects_instance);
        if (ret == nullptr) {
            Py_DECREF(ret);
            Py_DECREF(returned_objects_instance);
            set_error_info(semantics_error_class, use_python_error);
            return nullptr;
        }
    }
    if (PyObject_SetAttrString(returned_objects_instance, "outermost_inversion",
            outermost_inversion ? Py_True : Py_False) < 0) {
        Py_DECREF(returned_objects_instance);
        set_error_info(semantics_error_class, use_python_error);
        return nullptr;
    }
    auto ret_val = PyTuple_New(3);
    if (ret_val == nullptr) {
        set_error_info(PyExc_RuntimeError, use_python_error);
        return nullptr;
    }
    PyTuple_SetItem(ret_val, 0, returned_objects_instance);
    PyTuple_SetItem(ret_val, 1, PyUnicode_FromString(trial_text.c_str()));
    PyTuple_SetItem(ret_val, 2, PyUnicode_FromString(std::string(text).substr(trial_text.size()).c_str()));
    return ret_val;
#endif
#endif
}

static struct PyMethodDef spec_parser_functions[] =
{
    { "parse", (PyCFunction)parse, METH_VARARGS, docstr_parse },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef spec_parser_def =
{
    PyModuleDef_HEAD_INIT,
    "spec_parser",
    "Parse atom specifiers",
    -1,
    spec_parser_functions,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

    //RANGE_CHAR <- [^-#/:@,;()&| \t\n]
    //ATOM_NAME <- < [^#/:@; \t\n]+ >
static auto grammar = (R"---(
    atom_specifier <- as_term "&" atom_specifier / as_term "|" atom_specifier / as_term
    as_term <- "(" atom_specifier ")" zone_selector? / "~" as_term zone_selector? / SELECTOR_NAME zone_selector? / model_list
    model_list <- model+
    model <- HASH_TYPE model_hierarchy ("##" attribute_list)? model_parts* zone_selector? / "##" attribute_list model_parts* zone_selector? / model_parts zone_selector?
    model_hierarchy <- < model_range_list (!Space "." !Space model_hierarchy)* >
    model_range_list <- model_range ("," model_range_list)*
    model_range <- MODEL_SPEC_START "-" MODEL_SPEC_END / MODEL_SPEC_ANY
    model_parts <- chain+
    chain <- "/" part_list ("//" attribute_list)? chain_parts* / "//" attribute_list chain_parts* / chain_parts+
    chain_parts <- residue+
    residue <- ":" part_list ("::" attribute_list)? residue_parts* / "::" attribute_list residue_parts* / residue_parts+
    part_list <- PART_RANGE_LIST "," part_list / PART_RANGE_LIST
    residue_parts <- atom+
    # atom ranges are not allowed
    atom <- "@" atom_list ("@@" attribute_list)? / "@@" attribute_list
    atom_list <- ATOM_NAME "," atom_list / ATOM_NAME
    attribute_list <- attr_test ("," attr_test)*
    attr_test <- ATTR_NAME ATTR_OPERATOR ATTR_VALUE / "^" ATTR_NAME / ATTR_NAME
    zone_selector <- Space* ZONE_OPERATOR real_number / Space* ZONE_OPERATOR integer
    ATOM_NAME <- < [-+a-zA-Z0-9_'"*?\[\]\\]+ >
    ATTR_NAME <- < [a-zA-Z_] [a-zA-Z0-9_]* >
    ATTR_OPERATOR <- ">=" | ">" | "<=" | "<" | "==" | "=" | "!==" | "!=" | "<>"
    # Outer token delimiters to prevent automatic whitespace elimination inside quotes
    ATTR_VALUE <- < '"' < [^"]+ > '"' > / < "'" < [^']+ > "'" > / < [^#/:@,;"' ]+ >
    HASH_TYPE <- "#!" / "#"
    # limit model numbers to 5 digits to avoid conflicts with hex colors
    MODEL_SPEC <- < [0-9]{1,5} > ![0-9A-Fa-f]
    MODEL_SPEC_ANY <- MODEL_SPEC / "*"
    MODEL_SPEC_END <- MODEL_SPEC / "end" / "*"
    MODEL_SPEC_START <- MODEL_SPEC / "start" / "*"
    RANGE_CHAR <- [A-Za-z0-9_'"*?\[\]\\]
    RANGE_PART <- < "-"? RANGE_CHAR+ >
    PART_RANGE_LIST <- < RANGE_PART "-" RANGE_PART > / RANGE_PART
    SELECTOR_NAME <- < [a-zA-Z_][-+a-zA-Z0-9_]* >
    ZONE_OPERATOR <- "@>" | "@<" | ":>" | ":<" | "/>" | "/<" | "#>" | "#<"
    EndOfLine <- "\r\n" / "\n" / "\r"
    Space <- ' ' / '\t' / EndOfLine
    integer <- < [1-9][0-9]* >
    real_number <- < [0-9]* '.' [0-9]+ >
    %whitespace  <-  [ \t\r\n]*
)---");

#if 0
static bool py_bool_method(PyObject* py_model, PyObject* method_name)
{
    auto bool_result = PyObject_CallMethodNoArgs(py_model, method_name);
    if (bool_result == nullptr) {
        throw std::logic_error(use_python_error);
    }
    if (!PyBool_Check(bool_result)) {
        Py_DECREF(bool_result);
        throw std::domain_error("method did not return a boolean value");
    }
    // boolean constants are immortal, so don't sweat the DECREF
    return bool_result == Py_True;
}

static void add_model_to_objects(PyObject* py_model, PyObject* objects_inst)
{
    bool has_atoms;
    try {
        has_atoms = py_bool_method(py_model, atomspec_has_atoms_arg);
    } catch (std::domain_error& e) {
        Py_DECREF(objects_inst);
        throw std::logic_error("model.atomspec_has_atoms() did not return a boolean value");
    } catch (std::exception& e) {
        Py_DECREF(objects_inst);
        throw;
    }
    if (has_atoms) {
        auto method = PyObject_GetAttrString(py_model, "atomspec_atoms");
        if (method == nullptr) {
            Py_DECREF(objects_inst);
            throw std::logic_error(use_python_error);
        }
        if (!PyCallable_Check(method)) {
            Py_DECREF(objects_inst);
            Py_DECREF(method);
            throw std::logic_error("model.atomspec_atom is not callable");
        }
        auto no_args = PyTuple_New(0);
        if (no_args == nullptr)
            throw std::runtime_error("Cannot create zero-length tuple");
        auto kw_args = PyDict_New();
        if (kw_args == nullptr) {
            Py_DECREF(objects_inst);
            Py_DECREF(method);
            Py_DECREF(no_args);
            throw std::runtime_error("Cannot create keyword dictionary");
        }
        if (PyDict_SetItemString(kw_args, "ordered", order_implicit_atoms ? Py_True : Py_False) < 0) {
            Py_DECREF(objects_inst);
            Py_DECREF(method);
            Py_DECREF(no_args);
            Py_DECREF(kw_args);
            throw std::logic_error(use_python_error);
        }
        auto py_atoms = PyObject_Call(method, no_args, kw_args);
        if (PyObject_CallMethodOneArg(objects_inst, add_atoms_arg, py_atoms) == nullptr) {
            Py_DECREF(objects_inst);
            Py_DECREF(method);
            Py_DECREF(no_args);
            Py_DECREF(kw_args);
            Py_DECREF(py_atoms);
            throw std::logic_error(use_python_error);
        }
        Py_DECREF(method);
        Py_DECREF(no_args);
        Py_DECREF(kw_args);
        Py_DECREF(py_atoms);
    } else {
        bool has_pseudobonds;
        try {
            has_pseudobonds = py_bool_method(py_model, atomspec_has_pseudobonds_arg);
        } catch (std::domain_error& e) {
            Py_DECREF(objects_inst);
            throw std::logic_error(
                "model.atomspec_has_pseudobonds() did not return a boolean value");
        } catch (std::exception& e) {
            Py_DECREF(objects_inst);
            throw;
        }
        if (has_pseudobonds) {
            auto pbonds = PyObject_CallMethodNoArgs(py_model, atomspec_pseudobonds_arg);
            if (pbonds == nullptr) {
                Py_DECREF(objects_inst);
                throw std::logic_error(use_python_error);
            }
            auto ret_val = PyObject_CallMethodOneArg(objects_inst, add_pseudobonds_arg, pbonds);
            if (ret_val == nullptr) {
                Py_DECREF(objects_inst);
                throw std::logic_error(use_python_error);
            }
            Py_DECREF(ret_val);
        }
    }
    auto ret_val = PyObject_CallMethodOneArg(objects_inst, add_model_arg, py_model);
    if (ret_val == nullptr) {
        Py_DECREF(objects_inst);
        throw std::logic_error(use_python_error);
    }
    Py_DECREF(ret_val);
}

static void find_indices(const SemanticValues& vs, int& parts_index, int& attr_index, int& subparts_index)
{
    attr_index = parts_index = subparts_index = -1;
    if (vs.choice() == 0) {
        parts_index = 0;
        if (vs.size() == 2) {
            try {
                (void) std::any_cast<std::vector<AttrTester>>(vs[1]);
                attr_index = 1;
            } catch (std::bad_any_cast& e) {
                subparts_index = 1;
            }
        } else if (vs.size() == 3) {
            attr_index = 1;
            subparts_index = 2;
        }
    } else if (vs.choice() == 1) {
        attr_index = 0;
        if (vs.size() > 1)
            subparts_index = 1;
    } else
        subparts_index = 0;
}

static PyObject* get_module_attribute(const char* mod_name, const char* attr_name)
{
    auto mod = PyImport_ImportModule(mod_name);
    if (mod == nullptr)
        return nullptr;
    auto attr = PyObject_GetAttrString(mod, attr_name);
    Py_DECREF(mod);
    if (attr == nullptr)
        return nullptr;
    return attr;
}

static RGBA make_color_val(std::string& val_str)
{
    auto py_val_str = PyUnicode_FromString(val_str.c_str());
    auto color_val = PyObject_CallFunctionObjArgs(converter, session, py_val_str, nullptr);
    Py_DECREF(py_val_str);
    auto uint8x4 = PyObject_CallMethodNoArgs(color_val, uint8x4_arg);
    Py_DECREF(color_val);
    if (uint8x4 == nullptr)
        throw std::logic_error(use_python_error);
    if (!PySequence_Check(uint8x4)) {
        Py_DECREF(uint8x4);
        throw std::logic_error("Color.uint8x4() is not a sequence!");
    }
    if (PySequence_Length(uint8x4) != 4) {
        Py_DECREF(uint8x4);
        throw std::logic_error("Color.uint8x4() is not a sequence of 4 items!");
    }
    RGBA rgba;
    for (int i = 0; i < 4; ++i) {
        auto component = PySequence_GetItem(uint8x4, i);
        if (!PyNumber_Check(component)) {
            Py_DECREF(uint8x4);
            throw std::logic_error("Color.uint8x4() contains non-numbers!");
        }
        auto val = PyNumber_Long(component);
        if (val == nullptr) {
            Py_DECREF(uint8x4);
            throw std::logic_error("Color.uint8x4() contains non-integers!");
        }
        rgba.push_back(static_cast<unsigned char>(PyLong_AsLong(val)));
        Py_DECREF(val);
    }
    Py_DECREF(uint8x4);
    return rgba;
}

static PyObject*
process_model_attrs(PyObject* base_objects, const std::any& parser_model_attrs)
{
    auto model_attr_tests = std::any_cast<std::vector<AttrTester>>(parser_model_attrs);
    auto attrs_objects = new_objects_instance();
    auto obj_models = PyObject_GetAttrString(base_objects, "models");
    Py_DECREF(base_objects);
    if (obj_models == nullptr) {
        Py_DECREF(attrs_objects);
        throw std::logic_error(use_python_error);
    }
    auto fast_models = PySequence_Fast(obj_models, "Cannot convert Objects.models to fast sequence");
    Py_DECREF(obj_models);
    if (fast_models == nullptr) {
        Py_DECREF(attrs_objects);
        throw std::logic_error(use_python_error);
    }
    auto num_tests = model_attr_tests.size();
    auto py_attr_tests = PyTuple_New(num_tests);
    if (py_attr_tests == nullptr) {
        Py_DECREF(attrs_objects);
        Py_DECREF(fast_models);
        throw std::runtime_error("Could not create tuple");
    }
    for (decltype(num_tests) i = 0; i < num_tests; ++i) {
        auto py_attr_test = model_attr_tests[i].py_attr_test();
        Py_INCREF(py_attr_test);
        PyTuple_SET_ITEM(py_attr_tests, i, py_attr_test);
    }

    auto num_models = PySequence_Fast_GET_SIZE(fast_models);
    for (Py_ssize_t i = 0; i < num_models; ++i) {
        auto m = PySequence_Fast_GET_ITEM(fast_models, i);
        auto ret = PyObject_CallMethodOneArg(m, atomspec_model_attr_arg, py_attr_tests);
        if (ret == nullptr) {
            Py_DECREF(attrs_objects);
            Py_DECREF(fast_models);
            Py_DECREF(py_attr_tests);
            throw std::logic_error(use_python_error);
        }
        if (ret == Py_True) {
            try {
                add_model_to_objects(m, attrs_objects);
            } catch (std::exception& e) {
                Py_DECREF(fast_models);
                Py_DECREF(py_attr_tests);
                throw;
            }
        }
    }
    Py_DECREF(fast_models);
    Py_DECREF(py_attr_tests);
    return attrs_objects;
}

static PyObject*
vector_to_py_list(const std::any& parser_parts)
{
    auto parts = std::any_cast<std::vector<PyObject*>>(parser_parts);
    auto num_tests = parts.size();
    auto py_part_tests = PyList_New(num_tests);
    if (py_part_tests == nullptr) {
        throw std::runtime_error("Could not create tuple");
    }
    for (decltype(num_tests) i = 0; i < num_tests; ++i) {
        auto part = parts[i];
        PyList_SET_ITEM(py_part_tests, i, part);
    }
    return py_part_tests;
}

static PyObject*
process_model_parts(PyObject* base_objects, const std::any& parser_model_parts)
{
    auto py_part_tests = vector_to_py_list(parser_model_parts);

    auto parts_objects = new_objects_instance();
    auto obj_models = PyObject_GetAttrString(base_objects, "models");
    Py_DECREF(base_objects);
    if (obj_models == nullptr) {
        Py_DECREF(parts_objects);
        Py_DECREF(py_part_tests);
        throw std::logic_error(use_python_error);
    }

    auto fast_models = PySequence_Fast(obj_models, "Cannot convert Objects.models to fast sequence");
    Py_DECREF(obj_models);
    if (fast_models == nullptr) {
        Py_DECREF(parts_objects);
        Py_DECREF(py_part_tests);
        throw std::logic_error(use_python_error);
    }

    auto num_models = PySequence_Fast_GET_SIZE(fast_models);
    for (Py_ssize_t i = 0; i < num_models; ++i) {
        auto m = PySequence_Fast_GET_ITEM(fast_models, i);
        auto ret = PyObject_CallFunctionObjArgs(_add_model_parts, session, m, py_part_tests, parts_objects,
            order_implicit_atoms ? Py_True : Py_False, nullptr);
        if (ret == nullptr) {
            Py_DECREF(parts_objects);
            Py_DECREF(fast_models);
            Py_DECREF(py_part_tests);
            throw std::logic_error(use_python_error);
        }
        if (ret == Py_True) {
            try {
                add_model_to_objects(m, parts_objects);
            } catch (std::exception& e) {
                Py_DECREF(fast_models);
                Py_DECREF(py_part_tests);
                throw;
            }
        }
    }
    Py_DECREF(fast_models);
    Py_DECREF(py_part_tests);
    return parts_objects;
}

static PyObject*
process_zone(PyObject* base_objects, const std::any& parser_zone)
{
    auto zone_info = std::any_cast<std::pair<std::string_view, float>>(parser_zone);
    auto zone_objects = new_objects_instance();
    auto py_num_atoms = PyObject_GetAttrString(base_objects, "num_atoms");
    if (!PyLong_Check(py_num_atoms)) {
        Py_DECREF(base_objects);
        Py_DECREF(py_num_atoms);
        throw std::logic_error("Objects.num_atoms is not an integer!");
    }
    auto num_atoms = PyLong_AsLong(py_num_atoms);
    Py_DECREF(py_num_atoms);
    if (num_atoms > 0) {
        auto coords = PyObject_GetAttrString(
            PyObject_GetAttrString(base_objects, "atoms"), "scene_coords");
        auto list_size = PySequence_Fast_GET_SIZE(models);
        for (decltype(list_size) i = 0; i < list_size; ++i) {
            auto m = PySequence_Fast_GET_ITEM(models, i);
            auto ret = PyObject_CallMethod(m, "atomspec_zone", "OOfCCO", session, coords,
                zone_info.second, zone_info.first[0], zone_info.first[1], zone_objects);
            if (ret == nullptr) {
                Py_DECREF(base_objects);
                Py_DECREF(zone_objects);
                throw std::logic_error(use_python_error);
            }
        }
    }
    Py_DECREF(base_objects);
    return zone_objects;
}

static void
debug_semantic_values(const SemanticValues& vs)
{
    for (std::vector<std::any>::size_type item = 0; item < vs.size(); ++item)
        std::cerr << "semantic value " << item << (vs[item].has_value() ? " contains" : " does not contain") << " a value of type " << vs[item].type().name() << "\n";
}
#endif

PyMODINIT_FUNC PyInit__spec_parser()
{
    auto mod = PyModule_Create(&spec_parser_def);
    spec_parser.load_grammar(grammar);
    if (static_cast<bool>(spec_parser) == false) {
        PyErr_SetString(PyExc_SyntaxError, "Atom-specifier grammar is bad");
        return nullptr;
    }
#ifndef EVALUATE
    spec_parser.enable_ast();
#else
    // parsing "/V & protein" fails with packrat parsing on
    //spec_parser.enable_packrat_parsing();
    objects_class = get_module_attribute("chimerax.core.objects", "Objects");
    if (objects_class == nullptr)
        return nullptr;
    objects_intersect = PyObject_GetAttrString(objects_class, "intersect");
    if (objects_intersect == nullptr)
        return nullptr;
    objects_union = PyObject_GetAttrString(objects_class, "union");
    if (objects_union == nullptr)
        return nullptr;
    get_selector = get_module_attribute("chimerax.core.commands.atomspec", "get_selector");
    if (get_selector == nullptr)
        return nullptr;
    add_implied_bonds = get_module_attribute("chimerax.core.commands.atomspec", "add_implied_bonds");
    if (add_implied_bonds == nullptr)
        return nullptr;
    combine_arg = PyUnicode_FromString("combine");
    if (combine_arg == nullptr)
        return nullptr;
    invert_arg = PyUnicode_FromString("invert");
    if (invert_arg == nullptr)
        return nullptr;
    list_arg = PyUnicode_FromString("list");
    if (list_arg == nullptr)
        return nullptr;
    add_model_arg = PyUnicode_FromString("add_model");
    if (add_model_arg == nullptr)
        return nullptr;
    add_atoms_arg = PyUnicode_FromString("add_atoms");
    if (add_atoms_arg == nullptr)
        return nullptr;
    add_parts_arg = PyUnicode_FromString("add_parts");
    if (add_parts_arg == nullptr)
        return nullptr;
    add_pseudobonds_arg = PyUnicode_FromString("add_pseudobonds");
    if (add_pseudobonds_arg == nullptr)
        return nullptr;
    atomspec_has_atoms_arg = PyUnicode_FromString("atomspec_has_atoms");
    if (atomspec_has_atoms_arg == nullptr)
        return nullptr;
    atomspec_has_pseudobonds_arg = PyUnicode_FromString("atomspec_has_pseudobonds");
    if (atomspec_has_pseudobonds_arg == nullptr)
        return nullptr;
    atomspec_model_attr_arg = PyUnicode_FromString("atomspec_model_attr");
    if (atomspec_model_attr_arg == nullptr)
        return nullptr;
    atomspec_pseudobonds_arg = PyUnicode_FromString("atomspec_pseudobonds");
    if (atomspec_pseudobonds_arg == nullptr)
        return nullptr;
    ColorArg = get_module_attribute("chimerax.core.commands", "ColorArg");
    if (ColorArg == nullptr)
        return nullptr;
    make_converter = get_module_attribute("chimerax.core.commands", "make_converter");
    if (make_converter == nullptr)
        return nullptr;
    converter = PyObject_CallOneArg(make_converter, ColorArg);
    if (converter == nullptr)
        return nullptr;
    uint8x4_arg = PyUnicode_FromString("uint8x4");
    if (uint8x4_arg == nullptr)
        return nullptr;
    attr_test_class = get_module_attribute("chimerax.core.commands.atomspec", "_AttrTest");
    if (attr_test_class == nullptr)
        return nullptr;
    op_eq = get_module_attribute("operator", "eq");
    if (op_eq == nullptr)
        return nullptr;
    op_ne = get_module_attribute("operator", "ne");
    if (op_ne == nullptr)
        return nullptr;
    op_ge = get_module_attribute("operator", "ge");
    if (op_ge == nullptr)
        return nullptr;
    op_gt = get_module_attribute("operator", "gt");
    if (op_gt == nullptr)
        return nullptr;
    op_le = get_module_attribute("operator", "le");
    if (op_le == nullptr)
        return nullptr;
    op_lt = get_module_attribute("operator", "lt");
    if (op_lt == nullptr)
        return nullptr;
    op_not = get_module_attribute("operator", "not_");
    if (op_not == nullptr)
        return nullptr;
    op_truth = get_module_attribute("operator", "truth");
    if (op_truth == nullptr)
        return nullptr;
    _Chain_class = get_module_attribute("chimerax.core.commands.atomspec", "_Chain");
    if (_Chain_class == nullptr)
        return nullptr;
    _Residue_class = get_module_attribute("chimerax.core.commands.atomspec", "_Residue");
    if (_Residue_class == nullptr)
        return nullptr;
    _Atom_class = get_module_attribute("chimerax.core.commands.atomspec", "_Atom");
    if (_Atom_class == nullptr)
        return nullptr;
    _Part_class = get_module_attribute("chimerax.core.commands.atomspec", "_Part");
    if (_Part_class == nullptr)
        return nullptr;
    _PartList_class = get_module_attribute("chimerax.core.commands.atomspec", "_PartList");
    if (_PartList_class == nullptr)
        return nullptr;
    _add_model_parts = get_module_attribute("chimerax.core.commands.atomspec", "_add_model_parts");
    if (_add_model_parts == nullptr)
        return nullptr;

    // atom_specifier
    spec_parser["atom_specifier"] = [](const SemanticValues &vs) {
        PyObject* results;
        auto left_objects = std::any_cast<PyObject*>(vs[0]);
        if (vs.choice() == 2) {
            results = left_objects;
        } else {
            outermost_inversion = false;
            PyObject* op = vs.choice() == 0 ? objects_intersect : objects_union;
            auto right_objects = std::any_cast<PyObject*>(vs[1]);
            results = PyObject_CallFunctionObjArgs(op, left_objects, right_objects, nullptr);
            Py_DECREF(left_objects);
            Py_DECREF(right_objects);
            if (results == nullptr)
                throw std::logic_error(use_python_error);
        }
        return results;
    };

    // as_term
    spec_parser["as_term"] = [](const SemanticValues &vs) {
        auto objects_inst = std::any_cast<PyObject*>(vs[0]);
        if (vs.choice() == 1) {
            // tilde
            auto ret = PyObject_CallMethodObjArgs(objects_inst, invert_arg, session, models, nullptr);
            if (ret == nullptr)
                throw std::logic_error(use_python_error);
            Py_DECREF(ret);
            outermost_inversion = true;
        } else
            outermost_inversion = false;
        if (vs.size() == 1)
            return objects_inst;

        // there's a zone selector
        return process_zone(std::any_cast<PyObject*>(vs[0]), vs[1]);
    };

    // model_list
    spec_parser["model_list"] = [](const SemanticValues &vs) {
        PyObject* all_model_objects = nullptr;
        auto num_hierarchies = vs.size();
        for (decltype(num_hierarchies) i = 0; i < num_hierarchies; ++i) {
            auto model_objects = std::any_cast<PyObject*>(vs[i]);
            if (all_model_objects == nullptr)
                all_model_objects = model_objects;
            else {
                if (PyObject_CallMethodOneArg(all_model_objects, combine_arg, model_objects) == nullptr) {
                    Py_DECREF(model_objects);
                    throw std::logic_error(use_python_error);
                }
                Py_DECREF(model_objects);
            }
        }
        return all_model_objects;
    };
            
    // model <- HASH_TYPE model_hierarchy ("##" attribute_list)? model_parts* zone_selector? / "##" attribute_list model_parts* zone_selector? / model_parts zone_selector?
    spec_parser["model"] = [](const SemanticValues &vs) {
        auto objects_inst = new_objects_instance();
        //TODO: attrs, parts, zone
        int attr_index = -1, parts_index = -1, zone_index = -1;
        GlobalModelMatcher gmatcher;
        if (vs.choice() == 0) {
            gmatcher = GlobalModelMatcher(std::any_cast<bool>(vs[0]), 
                std::any_cast<std::vector<std::vector<ModelMatcher>>>(vs[1]));
            for (auto py_model: gmatcher.matches())
                add_model_to_objects(py_model, objects_inst);
        } else {
            auto list_size = PySequence_Fast_GET_SIZE(models);
            for (decltype(list_size) i = 0; i < list_size; ++i)
                add_model_to_objects(PySequence_Fast_GET_ITEM(models, i), objects_inst);
        }
        if (vs.choice() == 0) {
            if (vs.size() == 5) {
                attr_index = 2;
                parts_index = 3;
                zone_index = 4;
            } else if (vs.size() > 2) {
                for (std::vector<std::any>::size_type vs_i = 2; vs_i < vs.size(); ++vs_i) {
                    try {
                        (void) std::any_cast<std::vector<AttrTester>>(vs[vs_i]);
                        attr_index = vs_i;
                        continue;
                    } catch (std::bad_any_cast& e) {
                        ;
                    }
                    try {
                        (void) std::any_cast<std::pair<std::string_view, float>>(vs[vs_i]);
                        zone_index = vs_i;
                        continue;
                    } catch (std::bad_any_cast& e) {
                        ;
                    }
                    parts_index = vs_i;
                }
            }
        } else if (vs.choice() == 1) {
            attr_index = 0;
            if (vs.size() > 1)
                for (std::vector<std::any>::size_type vs_i = 1; vs_i < vs.size(); ++vs_i) {
                    try {
                        (void) std::any_cast<std::pair<std::string_view, float>>(vs[vs_i]);
                        zone_index = vs_i;
                        continue;
                    } catch (std::bad_any_cast& e) {
                        ;
                    }
                    parts_index = vs_i;
                }
            auto list_size = PySequence_Fast_GET_SIZE(models);
            for (decltype(list_size) i = 0; i < list_size; ++i) {
                auto m = PySequence_Fast_GET_ITEM(models, i);
                add_model_to_objects(m, objects_inst);
            }
        } else {
            parts_index = 0;
            if (vs.size() > 1)
                zone_index = 1;
        }
        if (attr_index >= 0)
            objects_inst = process_model_attrs(objects_inst, vs[attr_index]);
        if (parts_index >= 0)
            objects_inst = process_model_parts(objects_inst, vs[parts_index]);
        if (zone_index >= 0)
            objects_inst = process_zone(objects_inst, vs[zone_index]);
        return objects_inst;
    };
            
    // model_hierarchy
    spec_parser["model_hierarchy"] = [](const SemanticValues &vs) {
        std::vector<std::vector<ModelMatcher>> levels;
        levels.push_back(std::any_cast<std::vector<ModelMatcher>>(vs[0]));
        if (vs.size() == 2) {
            auto back_levels = std::any_cast<std::vector<std::vector<ModelMatcher>>>(vs[1]);
            levels.insert(levels.end(), back_levels.begin(), back_levels.end());
        }
        return levels;
    };
            
    // model_range_list
    spec_parser["model_range_list"] = [](const SemanticValues &vs) {
        std::vector<ModelMatcher> ranges;
        ranges.push_back(std::any_cast<ModelMatcher>(vs[0]));
        if (vs.size() == 2) {
            auto back_ranges = std::any_cast<std::vector<ModelMatcher>>(vs[1]);
            ranges.insert(ranges.end(), back_ranges.begin(), back_ranges.end());
        }
        return ranges;
    };
            
    // model_range
    spec_parser["model_range"] = [](const SemanticValues &vs) {
        if (vs.choice() == 0)
            return ModelMatcher(std::any_cast<ModelPart>(vs[0]), std::any_cast<ModelPart>(vs[1]));
        return ModelMatcher(std::any_cast<ModelPart>(vs[0]));
    };

    // model_parts <- chain+
    spec_parser["model_parts"] = [](const SemanticValues &vs) {
        std::vector<PyObject*> chains;
        for (auto v: vs)
            chains.push_back(std::any_cast<PyObject*>(v));
        return chains;
    };
            
    // chain <- "/" part_list ("//" attribute_list)? chain_parts* / "//" attribute_list chain_parts* / chain_parts+
    spec_parser["chain"] = [](const SemanticValues &vs) {
        int parts_index, attr_index, subparts_index;
        find_indices(vs, parts_index, attr_index, subparts_index);
        PyObject *part_list, *attr_list, *subpart_list;
        if (parts_index >= 0) {
            part_list = std::any_cast<PyObject*>(vs[parts_index]);
        } else
            part_list = Py_None;
        if (attr_index >= 0) {
            auto attr_tests = std::any_cast<std::vector<AttrTester>>(vs[attr_index]);
            auto at_size = attr_tests.size();
            attr_list = PyList_New(at_size);
            if (attr_list == nullptr) {
                Py_DECREF(part_list);
                throw std::runtime_error("Cannot create Python list for attribute tests");
            }
            for (decltype(at_size) i = 0; i < at_size; ++i) {
                auto py_attr_test = attr_tests[i].py_attr_test();
                Py_INCREF(py_attr_test);
                PyList_SET_ITEM(attr_list, i, py_attr_test);
            }
        } else
            attr_list = Py_None;
        if (subparts_index >= 0) {
            subpart_list = vector_to_py_list(vs[subparts_index]);
        } else
            subpart_list = Py_None;
        auto chain_part = PyObject_CallFunctionObjArgs(_Chain_class, part_list, attr_list, nullptr);
        if (part_list != Py_None)
            Py_DECREF(part_list);
        if (attr_list != Py_None)
            Py_DECREF(attr_list);
        if (subpart_list != Py_None) {
            if (chain_part != nullptr) {
                if (PyObject_SetAttrString(chain_part, "sub_parts", subpart_list) < 0) {
                    Py_DECREF(chain_part);
                    Py_DECREF(subpart_list);
                    throw std::logic_error(use_python_error);
                }
            }
            Py_DECREF(subpart_list);
        }
        if (chain_part == nullptr)
            throw std::logic_error(use_python_error);
        return chain_part;
    };
            
    // chain_parts <- residue+
    spec_parser["chain_parts"] = [](const SemanticValues &vs) {
        std::vector<PyObject*> residues;
        for (auto v: vs)
            residues.push_back(std::any_cast<PyObject*>(v));
        return residues;
    };
            
    // residue <- ":" part_list ("::" attribute_list)? residue_parts* / "::" attribute_list residue_parts* / residue_parts*
    spec_parser["residue"] = [](const SemanticValues &vs) {
        int parts_index, attr_index, subparts_index;
        find_indices(vs, parts_index, attr_index, subparts_index);
        PyObject *part_list, *attr_list, *subpart_list;
        if (parts_index >= 0) {
            part_list = std::any_cast<PyObject*>(vs[parts_index]);
        } else
            part_list = Py_None;
        if (attr_index >= 0) {
            auto attr_tests = std::any_cast<std::vector<AttrTester>>(vs[attr_index]);
            auto at_size = attr_tests.size();
            attr_list = PyList_New(at_size);
            if (attr_list == nullptr) {
                Py_DECREF(part_list);
                throw std::runtime_error("Cannot create Python list for attribute tests");
            }
            for (decltype(at_size) i = 0; i < at_size; ++i) {
                auto py_attr_test = attr_tests[i].py_attr_test();
                Py_INCREF(py_attr_test);
                PyList_SET_ITEM(attr_list, i, py_attr_test);
            }
        } else
            attr_list = Py_None;
        if (subparts_index >= 0) {
            subpart_list = vector_to_py_list(vs[subparts_index]);
        } else
            subpart_list = Py_None;
        auto residue_part = PyObject_CallFunctionObjArgs(_Residue_class, part_list, attr_list, nullptr);
        if (part_list != Py_None)
            Py_DECREF(part_list);
        if (attr_list != Py_None)
            Py_DECREF(attr_list);
        if (subpart_list != Py_None) {
            if (residue_part != nullptr) {
                if (PyObject_SetAttrString(residue_part, "sub_parts", subpart_list) < 0) {
                    Py_DECREF(residue_part);
                    Py_DECREF(subpart_list);
                    throw std::logic_error(use_python_error);
                }
            }
            Py_DECREF(subpart_list);
        }
        if (residue_part == nullptr)
            throw std::logic_error(use_python_error);
        return residue_part;
    };
            
    // residue_parts <- atom+
    spec_parser["residue_parts"] = [](const SemanticValues &vs) {
        std::vector<PyObject*> atoms;
        for (auto v: vs)
            atoms.push_back(std::any_cast<PyObject*>(v));
        return atoms;
    };

    // part_list
    spec_parser["part_list"] = [](const SemanticValues &vs) {
        PyObject* part_list;
        auto part = std::any_cast<PyObject*>(vs[0]);
        if (vs.choice() == 0) {
            part_list = std::any_cast<PyObject*>(vs[1]);
            if (PyObject_CallMethodOneArg(part_list, add_parts_arg, part) == nullptr) {
                Py_DECREF(part_list);
                Py_DECREF(part);
                throw std::logic_error(use_python_error);
            }
        } else {
            part_list = PyObject_CallFunctionObjArgs(_PartList_class, part, nullptr);
            if (part_list == nullptr) {
                Py_DECREF(part);
                throw std::logic_error(use_python_error);
            }
        }
        return part_list;
    };

    // atom <- "@" atom_list ("@@" attribute_list)? / "@@" attribute_list
    spec_parser["atom"] = [](const SemanticValues &vs) {
        int parts_index = -1, attr_index = -1;
        if (vs.choice() == 0) {
            parts_index = 0;
            if (vs.size() == 2)
                attr_index = 1;
        } else
            attr_index = 0;
        PyObject *part_list, *attr_list;
        if (parts_index >= 0) {
            part_list = std::any_cast<PyObject*>(vs[parts_index]);
        } else
            part_list = Py_None;
        if (attr_index >= 0) {
            auto attr_tests = std::any_cast<std::vector<AttrTester>>(vs[attr_index]);
            auto at_size = attr_tests.size();
            attr_list = PyList_New(at_size);
            if (attr_list == nullptr) {
                Py_DECREF(part_list);
                throw std::runtime_error("Cannot create Python list for attribute tests");
            }
            for (decltype(at_size) i = 0; i < at_size; ++i) {
                auto py_attr_test = attr_tests[i].py_attr_test();
                Py_INCREF(py_attr_test);
                PyList_SET_ITEM(attr_list, i, py_attr_test);
            }
        } else
            attr_list = Py_None;
        auto atom_part = PyObject_CallFunctionObjArgs(_Atom_class, part_list, attr_list, nullptr);
        if (part_list != Py_None)
            Py_DECREF(part_list);
        if (attr_list != Py_None)
            Py_DECREF(attr_list);
        if (atom_part == nullptr)
            throw std::logic_error(use_python_error);
        return atom_part;
    };
            
    // atom_list <- ATOM_NAME "," atom_list / ATOM_NAME
    spec_parser["atom_list"] = [](const SemanticValues &vs) {
        PyObject* part_list;
        auto part = std::any_cast<PyObject*>(vs[0]);
        if (vs.choice() == 0) {
            part_list = std::any_cast<PyObject*>(vs[1]);
            if (PyObject_CallMethodOneArg(part_list, add_parts_arg, part) == nullptr) {
                Py_DECREF(part_list);
                Py_DECREF(part);
                throw std::logic_error(use_python_error);
            }
        } else {
            part_list = PyObject_CallFunctionObjArgs(_PartList_class, part, nullptr);
            if (part_list == nullptr) {
                Py_DECREF(part);
                throw std::logic_error(use_python_error);
            }
        }
        return part_list;
    };
            
    // attribute_list
    spec_parser["attribute_list"] = [](const SemanticValues &vs) {
        std::vector<AttrTester> attr_tests;
        for (auto val: vs) {
            attr_tests.push_back(std::any_cast<AttrTester>(val));
        }
        return attr_tests;
    };

    // attr_test
    spec_parser["attr_test"] = [](const SemanticValues &vs) {
        // emulates the code in commands.atomspec._AtomSpecSemantics.attr_test
        if (vs.choice() == 0) {
            auto name = std::string(std::any_cast<std::string_view>(vs[0]));
            auto op = std::string(std::any_cast<std::string_view>(vs[1]));
            auto vstr_isquoted = std::any_cast<std::pair<std::string_view, bool>>(vs[2]);
            auto val_str = std::string(vstr_isquoted.first);
            auto quoted = vstr_isquoted.second;
            auto name_len = name.size();
            if (name_len >= 5 && name.substr(name_len-4) == "olor"
            && (name[name_len-5] == 'C' || name[name_len-5] == 'c')) {
                // val_str is color name
                return AttrTester(name, op, Value(make_color_val(val_str)));
            }
            if (quoted || op == "==" || op == "!==") {
                // quoted args are always strings, and case-sensitive compare must be a string
                return AttrTester(name, op, Value(val_str));
            }
            char* end_ptr;
            auto c_str = val_str.c_str();
            auto int_val = std::strtol(c_str, &end_ptr, 10);
            bool is_int = true;
            if (end_ptr == c_str)
                is_int = false;
            else {
                while (*end_ptr != '\0') {
                    if (!isspace(*end_ptr++)) {
                        is_int = false;
                        break;
                    }
                }
                if (is_int && (int_val == LONG_MAX || int_val == LONG_MIN))
                    throw std::invalid_argument("Integer value too large");
            }
            if (is_int)
                return AttrTester(name, op, Value(int_val));
            auto float_val = std::strtod(c_str, &end_ptr);
            bool is_float = true;
            if (end_ptr == c_str)
                is_float = false;
            else {
                while (*end_ptr != '\0') {
                    if (!isspace(*end_ptr++)) {
                        is_float = false;
                        break;
                    }
                }
                if (is_float && (float_val == HUGE_VALF || float_val == -HUGE_VALF))
                    throw std::invalid_argument("Floating-point value too large");
            }
            if (is_float)
                return AttrTester(name, op, Value(float_val));
            return AttrTester(name, op, Value(val_str));
        } else if (vs.choice() == 1)
            return AttrTester(std::any_cast<std::string_view>(vs[0]), false);
        return AttrTester(std::any_cast<std::string_view>(vs[0]), true);
    };

    // zone_selector
    spec_parser["zone_selector"] = [](const SemanticValues &vs) {
        float dist;
        if (vs.choice() == 0)
            dist = std::any_cast<float>(vs[vs.size()-1]);
        else
            dist = std::any_cast<int>(vs[vs.size()-1]);
        return std::make_pair(std::any_cast<std::string_view>(vs[vs.size()-2]), dist);
    };

    // ATOM_NAME <- < [^#/:@; \t\n]+ >
    spec_parser["ATOM_NAME"] = [](const SemanticValues &vs) {
        auto arg1 = PyUnicode_FromString(vs.token_to_string().c_str());
        if (arg1 == nullptr)
            throw std::runtime_error("Could not convert token to string");
        decltype(arg1) arg2 = Py_None;
        auto part = PyObject_CallFunctionObjArgs(_Part_class, arg1, arg2, nullptr);
        if (part == nullptr) {
            Py_DECREF(arg1);
            throw std::logic_error(use_python_error);
        }
        Py_DECREF(arg1);
        return part;
    };
            
    // ATTR_NAME
    spec_parser["ATTR_NAME"] = [](const SemanticValues &vs) {
        return vs.token();
    };
            
    // ATTR_OPERATOR
    spec_parser["ATTR_OPERATOR"] = [](const SemanticValues &vs) {
        return vs.token();
    };
            
    // ATTR_VALUE
    spec_parser["ATTR_VALUE"] = [](const SemanticValues &vs) {
        // Return pair with unquoted token and boolean indicating whether it was originally quoted;
        // quoted values have two vs.tokens() -- unquoted and quoted, so always return the first
        return std::make_pair(vs.tokens[0], vs.choice() < 2);
    };
            
    // HASH_TYPE
    spec_parser["HASH_TYPE"] = [](const SemanticValues &vs) {
        return vs.tokens[0][1] == '!';
    };
            
    // MODEL_SPEC
    spec_parser["MODEL_SPEC"] = [](const SemanticValues &vs) {
        return ModelPart(vs.token_to_number<int>());
    };
            
    // MODEL_SPEC_ANY
    spec_parser["MODEL_SPEC_ANY"] = [](const SemanticValues &vs) {
        if (vs.choice() == 1)
            return ModelPart();
        return std::any_cast<ModelPart>(vs[0]);
    };
            
    // MODEL_SPEC_START
    spec_parser["MODEL_SPEC_START"] = [](const SemanticValues &vs) {
        if (vs.choice() > 0)
            return ModelPart();
        return std::any_cast<ModelPart>(vs[0]);
    };
            
    // MODEL_SPEC_END
    spec_parser["MODEL_SPEC_END"] = [](const SemanticValues &vs) {
        if (vs.choice() > 0)
            return ModelPart();
        return std::any_cast<ModelPart>(vs[0]);
    };
            
    // RANGE_CHAR <- [A-Za-z0-9_'"\[\]\\]
    spec_parser["RANGE_CHAR"] = [](const SemanticValues &vs) {
        return vs.token();
    };
            
    // RANGE_PART <- < "-"? RANGE_CHAR+ >
    spec_parser["RANGE_PART"] = [](const SemanticValues &vs) {
        return vs.token_to_string();
    };

    // PART_RANGE_LIST <- < RANGE_PART "-" RANGE_PART > / RANGE_PART
    spec_parser["PART_RANGE_LIST"] = [](const SemanticValues &vs) {
        auto arg1 = PyUnicode_FromString(std::any_cast<std::string>(vs[0]).c_str());
        if (arg1 == nullptr)
            throw std::runtime_error("Could not convert token to string");
        decltype(arg1) arg2;
        if (vs.size() == 1)
            arg2 = Py_None;
        else {
            arg2 = PyUnicode_FromString(std::any_cast<std::string>(vs[1]).c_str());
            if (arg2 == nullptr)
                throw std::runtime_error("Could not convert token to string");
        }
        auto part = PyObject_CallFunctionObjArgs(_Part_class, arg1, arg2, nullptr);
        if (part == nullptr) {
            Py_DECREF(arg1);
            if (arg2 != Py_None)
                Py_DECREF(arg2);
            throw std::logic_error(use_python_error);
        }
        Py_DECREF(arg1);
        if (arg2 != Py_None)
            Py_DECREF(arg2);
        return part;
    };
            
    // SELECTOR_NAME
    spec_parser["SELECTOR_NAME"] = [](const SemanticValues &vs) {
        auto sel_text = PyUnicode_FromString(vs.token_to_string().c_str());
        if (sel_text == nullptr)
            throw std::runtime_error("Could not convert token to string");
        auto selector = PyObject_CallOneArg(get_selector, sel_text);
        if (selector == nullptr) {
            Py_DECREF(sel_text);
            throw std::logic_error(use_python_error);
        }
        Py_DECREF(sel_text);
        if (selector == Py_None) {
            Py_DECREF(selector);
            std::string err_msg = "\"";
            err_msg += vs.token_to_string();
            err_msg += "\" is not a selector name";
            err_valid = true;
            auto line_info = vs.line_info();
            err_line = line_info.first;
            err_col = line_info.second;
            throw std::invalid_argument(err_msg);
        }
        auto is_inst = PyObject_IsInstance(selector, objects_class);
        if (is_inst < 0) {
            Py_DECREF(selector);
            throw std::runtime_error(use_python_error);
        }
        auto selector_objects = new_objects_instance();
        if (is_inst) {
            if (PyObject_CallMethodOneArg(selector_objects, combine_arg, selector) == nullptr) {
                Py_DECREF(selector);
                Py_DECREF(selector_objects);
                throw std::logic_error(use_python_error);
            }
            Py_DECREF(selector);
        } else {
            auto args = PyTuple_New(3);
            if (args == nullptr) {
                Py_DECREF(selector);
                Py_DECREF(selector_objects);
                throw std::runtime_error("Could not create 3-tuple for selector args");
            }
            PyTuple_SetItem(args, 0, session);
            Py_INCREF(session);
            PyTuple_SetItem(args, 1, models);
            Py_INCREF(models);
            PyTuple_SetItem(args, 2, selector_objects);
            Py_INCREF(selector_objects);
            auto ret = PyObject_CallObject(selector, args);
            Py_DECREF(selector);
            Py_DECREF(args);
            if (ret == nullptr) {
                Py_DECREF(selector_objects);
                throw std::logic_error(use_python_error);
            }
            Py_DECREF(ret);
        }
        return selector_objects;
    };
            
    // ZONE_OPERATOR
    spec_parser["ZONE_OPERATOR"] = [](const SemanticValues &vs) {
        return vs.token();
    };

    // integer
    spec_parser["integer"] = [](const SemanticValues &vs) {
        return vs.token_to_number<int>();
    };

    // real_number
    spec_parser["real_number"] = [](const SemanticValues &vs) {
        return vs.token_to_number<float>();
    };
#endif

    return mod;
}
