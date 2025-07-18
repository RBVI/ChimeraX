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

#include <algorithm> // std::transform
#include <cctype> // std::tolower
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>

#include "Python.h"
#include "peglib.h"

//#include <logger/logger.h>

using namespace peg;

static parser spec_parser;
static PyObject* session;
static PyObject *parse_error_class, *semantics_error_class;
static PyObject* add_part_arg;
static PyObject* add_parts_arg;
static PyObject* append_arg;
static PyObject* uint8x4_arg;
static PyObject* AtomSpec_class;
static PyObject* ColorArg_class;
static PyObject* TinyArray_class;
static PyObject* UserError_class;
static PyObject* _Atom_class;
static PyObject* _AttrList_class;
static PyObject* _AttrTest_class;
static PyObject* _Chain_class;
static PyObject* _Invert_class;
static PyObject* _Model_class;
static PyObject* _ModelHierarchy_class;
static PyObject* _ModelList_class;
static PyObject* _ModelRange_class;
static PyObject* _ModelRangeList_class;
static PyObject* _Part_class;
static PyObject* _PartList_class;
static PyObject* _Residue_class;
static PyObject* _SelectorName_class;
static PyObject* _Term_class;
static PyObject* _ZoneSelector_class;
static PyObject* color_converter_func;
static PyObject* get_selector_func;
static PyObject* op_eq;
static PyObject* op_ge;
static PyObject* op_gt;
static PyObject* op_le;
static PyObject* op_lt;
static PyObject* op_ne;
static PyObject* op_not;
static PyObject* op_truth;
static PyObject* end_string;
static PyObject* star_string;
static PyObject* start_string;
static std::string use_python_error("Use Python error");
static bool add_implied;

class SemanticsError: public std::logic_error {
public:
    SemanticsError(std::string& msg) : std::logic_error(msg) {}
};
class UserError: public std::invalid_argument {
public:
    UserError(std::string& msg) : std::invalid_argument(msg) {}
};

static const char*
docstr_parse = \
"parse(session, text)\n" \
"\n" \
"Parse the given text for an initial atom spec and if one is found" \
" return an chimera.core.commands.atomspec.AtomSpec instance," \
" the part of text used for the atom spec, and the remaining text.";

static size_t err_line, err_col;
static std::string err_msg;
static bool err_valid;

static void
set_error_info(PyObject* err_type, std::string msg)
{
    auto err_val = PyList_New(2);
    if (err_val == nullptr) {
        PyErr_SetString(PyExc_AssertionError, "Could not create error-value list");
        throw std::runtime_error("Could not create tuple");
    }
    PyList_SetItem(err_val, 0, PyLong_FromSize_t( err_valid ? err_col-1 : (size_t)0));
    if (PyErr_Occurred() == nullptr) {
        if (msg == use_python_error) {
            PyErr_SetString(PyExc_AssertionError, "Trying to use Python error when none set");
            throw std::runtime_error("No Python error message to use");
        }
        PyList_SetItem(err_val, 1, PyUnicode_FromString(msg.c_str()));
        PyErr_SetObject(err_type, err_val);
    } else {
        PyObject *type, *value, *traceback;
        //TODO: once ChimeraX at 3.12+, switch to PyErr_GetRaisedException
        PyErr_Fetch(&type, &value, &traceback);
        if (msg == use_python_error) {
            PyList_SetItem(err_val, 1, value);
        } else {
            PyList_SetItem(err_val, 1, PyUnicode_FromString(msg.c_str()));
            Py_DECREF(value);
        }
        //TODO: once ChimeraX at 3.12+, switch to PyErr_SetRaisedException
        PyErr_Restore(err_type, err_val, traceback);
        Py_DECREF(type);
    }
}

// Fixed text strings that need to be handed off to the Python layer
enum Symbols { OP_UNION, OP_INTERSECT, NUM_SYMBOLS };
static std::vector<std::string> symbols = { "&", "|" };
static std::vector<PyObject*> py_symbols;

void
print_ast(const Ast & ast)
{
    std::cerr << "  " << ast.name << " '" << ast.token_to_string() << "'; choice " << ast.choice << "  " << ast.nodes.size() << " subnodes\n";
    for (auto node: ast.nodes) {
        print_ast(*node);
    }
}
    
// NOTE: to figure out how things work in the Python layer you have to not only look at the
// corresponding parsing class, but also the corresponding method of _AtomSpecSemantics

static PyObject* eval_atom_spec(const Ast &ast);

static PyObject*
eval_integer(const Ast &ast) {
std::cerr << "eval_integer\n";
    // integer <- < [1-9][0-9]* >
    auto integer = PyNumber_Long(PyUnicode_FromString(ast.token_to_string().c_str()));
    if (integer == nullptr)
        throw SemanticsError(use_python_error);
    return integer;
}

static PyObject*
eval_real_number(const Ast &ast) {
std::cerr << "eval_real_number\n";
    // real_number <- < [0-9]* '.' [0-9]+ >
    auto real = PyNumber_Float(PyUnicode_FromString(ast.token_to_string().c_str()));
    if (real == nullptr)
        throw SemanticsError(use_python_error);
    return real;
}

static PyObject*
eval_ZONE_OPERATOR(const Ast &ast) {
std::cerr << "eval_ZONE_OPERATOR\n";
    // ZONE_OPERATOR <- "@>" | "@<" | ":>" | ":<" | "/>" | "/<" | "#>" | "#<"
    return PyUnicode_FromString(ast.token_to_string().c_str());
}

static PyObject*
eval_zone_selector(const Ast &ast) {
std::cerr << "eval_zone_selector\n";
    // zone_selector <- ZONE_OPERATOR _ real_number / ZONE_OPERATOR _ integer
    auto zone_op = eval_ZONE_OPERATOR(*ast.nodes[0]);
    auto distance = (ast.choice == 0 ? eval_real_number : eval_integer)(*ast.nodes[1]);
    auto zone_sel = PyObject_CallFunctionObjArgs(_ZoneSelector_class, zone_op, distance, nullptr);
    if (zone_sel == nullptr) {
        Py_DECREF(distance);
        throw SemanticsError(use_python_error);
    }
    return zone_sel;
}

static std::pair<std::string, PyObject*>
eval_ATTR_NAME(const Ast &ast) {
    // ATTR_NAME <- < [a-zA-Z_] [a-zA-Z0-9_]* >
std::cerr << "eval_ATTR_NAME\n";
    auto token = ast.token_to_string();
    return std::pair<std::string, PyObject*>(token, PyUnicode_FromString(token.c_str()));
}

static std::pair<std::string, PyObject*>
eval_ATTR_OPERATOR(const Ast &ast) {
std::cerr << "eval_ATTR_OPERATOR\n";
    // ATTR_OPERATOR <- ">=" | ">" | "<=" | "<" | "==" | "=" | "!==" | "!=" | "<>"
    auto token = ast.token_to_string();
    PyObject *op;
    if (token == "=")
        op = op_eq;
    else if (token == "!=")
        op = op_ne;
    else if (token == ">=")
        op = op_ge;
    else if (token == ">")
        op = op_gt;
    else if (token == "<=")
        op = op_le;
    else if (token == "<")
        op = op_lt;
    else
        op = PyUnicode_FromString(token.c_str());
    return std::pair<std::string, PyObject*>(token, op);
}

static PyObject*
eval_ATTR_VALUE(const Ast &ast, std::string& attr_name, std::string& op) {
std::cerr << "eval_ATTR_VALUE\n";
    // ATTR_VALUE <- < '"' < [^"]+ > '"' > / < "'" < [^']+ > "'" > / < [^#/:@,;"' ]+ >
    PyObject* value;
    auto vstr = ast.token_to_string();
    // quoted values need to always be treated as strings
    auto quoted = ast.choice < 2;
    auto lower_attr_name = attr_name;
    std::transform(lower_attr_name.begin(), lower_attr_name.end(), lower_attr_name.begin(),
        [](unsigned char c){ return std::tolower(c); });
    if (lower_attr_name.substr(std::max(0, static_cast<int>(lower_attr_name.size())-5)) == "color") {
        // if attr name ends with color, convert to color
        auto color = PyObject_CallFunctionObjArgs(color_converter_func, session,
            PyUnicode_FromString(vstr.c_str()), nullptr);
        if (color == nullptr) {
            PyObject *type, *err_value, *traceback;
            //TODO: once ChimeraX at 3.12+, switch to PyErr_GetRaisedException
            PyErr_Fetch(&type, &err_value, &traceback);
            std::string msg;
            msg.append("bad color: ");
            msg.append(vstr);
            if (PyUnicode_Check(err_value)) {
                msg.append(": ");
                msg.append(PyUnicode_AsUTF8(err_value));
            }
            //TODO: once ChimeraX at 3.12+, switch to PyErr_SetRaisedException
            PyErr_Restore(type, err_value, traceback);
            throw UserError(msg);
        }
        auto color_array = PyObject_CallMethodNoArgs(color, uint8x4_arg);
        if (color_array == nullptr) {
            Py_DECREF(color);
            throw SemanticsError(use_python_error);
        }
        value = PyObject_CallOneArg(TinyArray_class, color_array);
        if (value == nullptr) {
            Py_DECREF(color);
            Py_DECREF(color_array);
            throw SemanticsError(use_python_error);
        }
    } else if (quoted || op == "==" || op == "!==") {
        // case-sensitive compare must be a string
        value = PyUnicode_FromString(vstr.c_str());
    } else {
        // convert to best matching common type
        value = PyUnicode_FromString(vstr.c_str());
        auto int_value = PyNumber_Long(value);
        if (int_value == nullptr) {
            PyErr_Clear();
            auto float_value = PyNumber_Float(value);
            if (float_value == nullptr) {
                PyErr_Clear();
            } else {
                Py_DECREF(value);
                value = float_value;
            }
        } else {
            Py_DECREF(value);
            value = int_value;
        }
    }
    return value;
}

static PyObject*
eval_attr_test(const Ast &ast) {
std::cerr << "eval_attr_test\n";
    // attr_test <- ATTR_NAME ATTR_OPERATOR ATTR_VALUE / "^" ATTR_NAME / ATTR_NAME
    auto str_obj = eval_ATTR_NAME(*ast.nodes[0]);
    // replicate logic of chimerax.core.commands.atomspec._AtomSpecSemantics.attr_test
    PyObject* op;
    PyObject* value;
    if (ast.choice == 0) {
        auto str_op = eval_ATTR_OPERATOR(*ast.nodes[1]);
        op = str_op.second;
        value = eval_ATTR_VALUE(*ast.nodes[2], str_obj.first, str_op.first);
    } else if (ast.choice == 1) {
        op = op_not;
        value = Py_None;
    } else {
        op = op_truth;
        value = Py_None;
    }
    auto attr_test = PyObject_CallFunctionObjArgs(_AttrTest_class, ast.choice == 1 ? Py_True : Py_None,
        str_obj.second, op, value, nullptr);
    if (attr_test == nullptr)
        throw SemanticsError(use_python_error);
    return attr_test;
}

static PyObject*
eval_attribute_list(const Ast &ast) {
std::cerr << "eval_attribute_list\n";
    // attribute_list <- attr_test ("," _ attr_test)*
    auto attr_list = PyObject_CallNoArgs(_AttrList_class);
    if (attr_list == nullptr)
        throw SemanticsError(use_python_error);
    for (auto node: ast.nodes) {
        if (PyObject_CallMethodOneArg(attr_list, append_arg, eval_attr_test(*node)) == nullptr) {
            Py_DECREF(attr_list);
            throw SemanticsError(use_python_error);
        }
    }
    return attr_list;
}

static PyObject*
eval_PART_RANGE_LIST(const Ast &ast) {
std::cerr << "eval_PART_RANGE_LIST\n";
    // PART_RANGE_LIST <- < RANGE_PART _ "-" _ RANGE_PART > / RANGE_PART
    PyObject* start;
    PyObject* end;
    auto token = ast.token_to_string();
    if (ast.choice == 0) {
        auto dash_index = token.find_first_of('-', 1);
        start = PyUnicode_FromString(token.substr(0, dash_index).c_str());
        end = PyUnicode_FromString(token.substr(dash_index+1).c_str());
    } else {
        start = PyUnicode_FromString(token.c_str());
        end = Py_None;
    }
    auto part = PyObject_CallFunctionObjArgs(_Part_class, start, end, nullptr);
    if (part == nullptr)
        throw SemanticsError(use_python_error);
    return part;
}

static PyObject*
eval_part_list(const Ast &ast) {
std::cerr << "eval_part_list\n";
    // part_list <- PART_RANGE_LIST "," part_list / PART_RANGE_LIST
    auto part_range = eval_PART_RANGE_LIST(*ast.nodes[0]);
    PyObject* part_list;
    if (ast.choice == 0) {
        part_list = eval_part_list(*ast.nodes[1]);
        if (PyObject_CallMethodOneArg(part_list, add_parts_arg, part_range) == nullptr) {
            Py_DECREF(part_list);
            throw SemanticsError(use_python_error);
        }
    } else {
        part_list = PyObject_CallFunctionObjArgs(_PartList_class, part_range, nullptr);
        if (part_list == nullptr)
            throw SemanticsError(use_python_error);
    }
    return part_list;
}

static PyObject*
eval_ATOM_NAME(const Ast &ast) {
std::cerr << "eval_ATOM_NAME\n";
    // ATOM_NAME <- < [-+a-zA-Z0-9_'"*?\[\]\\]+ >
    auto part = PyObject_CallFunctionObjArgs(_Part_class,
        PyUnicode_FromString(ast.token_to_string().c_str()), Py_None, nullptr);
    if (part == nullptr)
        throw SemanticsError(use_python_error);
    return part;
}

static PyObject*
eval_atom_list(const Ast &ast) {
std::cerr << "eval_atom_list\n";
    // atom_list <- ATOM_NAME "," atom_list / ATOM_NAME
    auto atom_name = eval_ATOM_NAME(*ast.nodes[0]);
    PyObject* atom_list;
    if (ast.choice == 0) {
        atom_list = eval_atom_list(*ast.nodes[1]);
        if (PyObject_CallMethodOneArg(atom_list, add_parts_arg, atom_name) == nullptr) {
            Py_DECREF(atom_list);
            throw SemanticsError(use_python_error);
        }
    } else {
        atom_list = PyObject_CallFunctionObjArgs(_PartList_class, atom_name, nullptr);
        if (atom_list == nullptr)
            throw SemanticsError(use_python_error);
    }
    return atom_list;
}

static PyObject*
eval_atom(const Ast &ast) {
std::cerr << "eval_atom\n";
    // atom <- "@" atom_list ("@@" attribute_list)? / "@@" attribute_list
    PyObject* attrs = Py_None;
    PyObject* atom_list = Py_None;
    for (auto node: ast.nodes) {
        if (node->name == "atom_list") {
            atom_list = eval_atom_list(*node);
        } else if (node->name == "attribute_list") {
            attrs = eval_attribute_list(*node);
        }
    }
    auto atom = PyObject_CallFunctionObjArgs(_Atom_class, atom_list, attrs, nullptr);
    if (atom == nullptr)
        throw SemanticsError(use_python_error);
    return atom;
}

static std::vector<PyObject*>
eval_residue_parts(const Ast &ast) {
std::cerr << "eval_residue_parts\n";
    // residue_parts <- atom+
    std::vector<PyObject*> atoms;
    for (auto node: ast.nodes) {
        atoms.push_back(eval_atom(*node));
    }
    return atoms;
}

static PyObject*
eval_residue(const Ast &ast) {
std::cerr << "eval_residue\n";
    // residue <- ":" part_list ("::" attribute_list)? residue_parts* / "::" attribute_list residue_parts* / residue_parts+
    PyObject* attrs = Py_None;
    PyObject* part_list = Py_None;
    std::vector<PyObject*> parts;
    for (auto node: ast.nodes) {
        if (node->name == "part_list") {
            part_list = eval_part_list(*node);
        } else if (node->name == "attribute_list") {
            attrs = eval_attribute_list(*node);
        } else if (node->name == "residue_parts") {
            parts = eval_residue_parts(*node);
        }
    }
    // part_list and attribute_list go in _Residue constructor; residue_parts use .add_part()
    auto residue = PyObject_CallFunctionObjArgs(_Residue_class, part_list, attrs, nullptr);
    if (residue == nullptr)
        throw SemanticsError(use_python_error);
    for (auto part: parts) {
        if (PyObject_CallMethodOneArg(residue, add_part_arg, part) == nullptr) {
            Py_DECREF(residue);
            throw SemanticsError(use_python_error);
        }
    }
    return residue;
}

static std::vector<PyObject*>
eval_chain_parts(const Ast &ast) {
std::cerr << "eval_chain_parts\n";
    // chain_parts <- residue+
    std::vector<PyObject*> residues;
    for (auto node: ast.nodes) {
        residues.push_back(eval_residue(*node));
    }
    return residues;
}

static PyObject*
eval_chain(const Ast &ast) {
std::cerr << "eval_chain\n";
    // chain <- "/" part_list ("//" attribute_list)? chain_parts* / "//" attribute_list chain_parts* / chain_parts+
    PyObject* attrs = Py_None;
    PyObject* part_list = Py_None;
    std::vector<PyObject*> parts;
    for (auto node: ast.nodes) {
        if (node->name == "part_list") {
            part_list = eval_part_list(*node);
        } else if (node->name == "attribute_list") {
            attrs = eval_attribute_list(*node);
        } else if (node->name == "chain_parts") {
            parts = eval_chain_parts(*node);
        }
    }
    // part_list and attribute_list go in _Chain constructor; chain_parts use .add_part()
    auto chain = PyObject_CallFunctionObjArgs(_Chain_class, part_list, attrs, nullptr);
    if (chain == nullptr)
        throw SemanticsError(use_python_error);
    for (auto part: parts) {
        if (PyObject_CallMethodOneArg(chain, add_part_arg, part) == nullptr) {
            Py_DECREF(chain);
            throw SemanticsError(use_python_error);
        }
    }
    return chain;
}

static std::vector<PyObject*>
eval_model_parts(const Ast &ast) {
std::cerr << "eval_model_parts\n";
    // model_parts <- chain+
    std::vector<PyObject*> chains;
    for (auto node: ast.nodes) {
        chains.push_back(eval_chain(*node));
    }
    return chains;
}

static PyObject*
eval_MODEL_SPEC(const Ast &ast) {
std::cerr << "eval_MODEL_SPEC\n";
    // MODEL_SPEC <- < [0-9]{1,5} > ![0-9A-Fa-f]
    return PyLong_FromLong(ast.token_to_number<long>());
}

static PyObject*
eval_MODEL_SPEC_START(const Ast &ast) {
std::cerr << "eval_MODEL_SPEC_START\n";
    // MODEL_SPEC_START <- MODEL_SPEC / "start" / "*"
    if (ast.choice == 0)
        return eval_MODEL_SPEC(*ast.nodes[0]);
    return ast.choice == 1 ? start_string : star_string;
}

static PyObject*
eval_MODEL_SPEC_END(const Ast &ast) {
std::cerr << "eval_MODEL_SPEC_END\n";
    // MODEL_SPEC_END <- MODEL_SPEC / "end" / "*"
    if (ast.choice == 0)
        return eval_MODEL_SPEC(*ast.nodes[0]);
    return ast.choice == 1 ? end_string : star_string;
}

static PyObject*
eval_MODEL_SPEC_ANY(const Ast &ast) {
std::cerr << "eval_MODEL_SPEC_ANY\n";
    // MODEL_SPEC_ANY <- MODEL_SPEC / "*"
    if (ast.choice == 0)
        return eval_MODEL_SPEC(*ast.nodes[0]);
    return star_string;
}

static PyObject*
eval_model_range(const Ast &ast) {
std::cerr << "eval_model_range\n";
    // model_range <- MODEL_SPEC_START _ "-" _ MODEL_SPEC_END / MODEL_SPEC_ANY
    PyObject* left;
    PyObject* right;
    if (ast.choice == 0) {
        left = eval_MODEL_SPEC_START(*ast.nodes[0]);
        right = eval_MODEL_SPEC_END(*ast.nodes[1]);
    } else {
        left = eval_MODEL_SPEC_ANY(*ast.nodes[0]);
        right = Py_None;
    }
    auto mr = PyObject_CallFunctionObjArgs(_ModelRange_class, left, right, nullptr);
    if (mr == nullptr)
        throw SemanticsError(use_python_error);
    return mr;
}

static PyObject*
eval_model_range_list(const Ast &ast) {
std::cerr << "eval_model_range_list\n";
    // model_range_list <- model_range ("," _ model_range)*
    auto range = eval_model_range(*ast.nodes[0]);
    auto mrl = PyObject_CallFunctionObjArgs(_ModelRangeList_class, range, nullptr);
    if (mrl == nullptr)
        throw SemanticsError(use_python_error);
    for (auto i = 1u; i < ast.nodes.size(); ++i) {
        range = eval_model_range(*ast.nodes[i]);
        if (PyObject_CallMethodOneArg(mrl, append_arg, range) == nullptr) {
            Py_DECREF(mrl);
            throw SemanticsError(use_python_error);
        }
    }
    return mrl;
}

static PyObject*
eval_model_hierarchy(const Ast &ast) {
std::cerr << "eval_model_hierarchy\n";
    // model_hierarchy <- model_range_list ("." model_range_list)*
    auto rl = eval_model_range_list(*ast.nodes[0]);
    auto hierarchy = PyObject_CallFunctionObjArgs(_ModelHierarchy_class, rl, nullptr);
    if (hierarchy == nullptr)
        throw SemanticsError(use_python_error);
    for (auto i = 1u; i < ast.nodes.size(); ++i) {
        rl = eval_model_range_list(*ast.nodes[i]);
        if (PyObject_CallMethodOneArg(hierarchy, append_arg, rl) == nullptr) {
            Py_DECREF(hierarchy);
            throw SemanticsError(use_python_error);
        }
    }
    return hierarchy;
}

static PyObject*
eval_model(const Ast &ast) {
std::cerr << "eval_model\n";
    // model <- HASH_TYPE _ model_hierarchy (_ "##" _ attribute_list)? _ model_parts* _ zone_selector? / "##" _ attribute_list _ model_parts* _ zone_selector? / model_parts _ zone_selector?
    bool exact_match = false;
    PyObject* hierarchy = Py_None;
    PyObject* attrs = Py_None;
    std::vector<PyObject*> parts;
    PyObject* zone = Py_None;
    for (auto node: ast.nodes) {
        if (node->name == "HASH_TYPE") {
            // HASH_TYPE <- "#!" / "#"
            exact_match = node->choice == 0;
        } else if (node->name == "model_hierarchy") {
            hierarchy = eval_model_hierarchy(*node);
        } else if (node->name == "attribute_list") {
            attrs = eval_attribute_list(*node);
        } else if (node->name == "model_parts") {
            parts = eval_model_parts(*node);
        } else if (node->name == "zone_selector") {
            zone = eval_zone_selector(*node);
        }
    }
    auto model = PyObject_CallFunctionObjArgs(_Model_class, (exact_match ? Py_True : Py_False),
        hierarchy, attrs, nullptr);
    if (model == nullptr)
        throw SemanticsError(use_python_error);
    for (auto part: parts) {
        if (PyObject_CallMethodOneArg(model, add_part_arg, part) == nullptr) {
            Py_DECREF(model);
            throw SemanticsError(use_python_error);
        }
    }
    if (zone == Py_None)
        return model;
    
    if (PyObject_SetAttrString(zone, "model", model) < 0) {
        Py_DECREF(model);
        throw SemanticsError(use_python_error);
    }
    return zone;
}

static PyObject*
eval_model_list(const Ast &ast) {
std::cerr << "eval_model_list\n";
    // model_list <- model+
    auto py_ml = PyObject_CallNoArgs(_ModelList_class);
    for (auto node: ast.nodes) {
        auto model = eval_model(*node);
        if (model == nullptr) {
            Py_DECREF(py_ml);
            throw SemanticsError(use_python_error);
        }
        if (PyObject_CallMethodOneArg(py_ml, append_arg, model) == nullptr) {
            Py_DECREF(py_ml);
            Py_DECREF(model);
            throw SemanticsError(use_python_error);
        }
    }
    return py_ml;
}

static PyObject*
eval_SELECTOR_NAME(const Ast &ast) {
std::cerr << "eval_SELECTOR_NAME\n";
    // SELECTOR_NAME <- < [a-zA-Z_][-+a-zA-Z0-9_]* >
    auto token = ast.token_to_string();
    PyObject* name = PyUnicode_FromString(token.c_str());
    auto result = PyObject_CallOneArg(get_selector_func, name);
    if (result == nullptr)
        throw SemanticsError(use_python_error);
    if (result == Py_None) {
        std::string msg;
        msg.push_back('"');
        msg.append(token);
        msg.push_back('"');
        msg.append(" is not a selector name");
        throw SemanticsError(msg);
    }
    auto sn = PyObject_CallOneArg(_SelectorName_class, name);
    if (sn == nullptr)
        throw SemanticsError(use_python_error);
    return sn;
}

static PyObject*
eval_as_term(const Ast &ast) {
std::cerr << "eval_as_term\n";
    // as_term <- "(" _ atom_specifier _ ")" _ zone_selector? / "~" _ as_term _ zone_selector? / SELECTOR_NAME _ zone_selector? / model_list
    PyObject* as_term;
    PyObject* inner_as_term;
    PyObject* args;
    PyObject* kw_args;
    switch (ast.choice) {
        case 0:
            as_term = eval_atom_spec(*ast.nodes[0]);
            break;
        case 1:
            // tilde
            inner_as_term = eval_as_term(*ast.nodes[0]);

            args = PyTuple_New(1);
            if (args == nullptr) {
                throw std::runtime_error("Could not create tuple");
            }
            PyTuple_SET_ITEM(args, 0, inner_as_term);

            kw_args = PyDict_New();
            if (kw_args == nullptr) {
                Py_DECREF(args);
                throw std::runtime_error("Cannot create keyword dictionary");
            }
            if (PyDict_SetItemString(kw_args, "add_implied", add_implied ? Py_True : Py_False) < 0) {
                Py_DECREF(args);
                Py_DECREF(kw_args);
                throw SemanticsError(use_python_error);
            }

            as_term = PyObject_Call(_Invert_class, args, kw_args);
            if (as_term == nullptr) {
                Py_DECREF(inner_as_term);
                throw SemanticsError(use_python_error);
            }
            break;
        case 2:
            // selector
            as_term = PyObject_CallOneArg(_Term_class, eval_SELECTOR_NAME(*ast.nodes[0]));
            break;
        case 3:
            return PyObject_CallOneArg(_Term_class, eval_model_list(*ast.nodes[0]));
    }
    if (ast.nodes.size() > 1) {
        auto zone = eval_zone_selector(*ast.nodes[1]);
        if (PyObject_SetAttrString(zone, "model", as_term) < 0) {
            Py_DECREF(as_term);
            throw SemanticsError(use_python_error);
        }
        return zone;
    }
    return as_term;
}
    
static PyObject*
eval_atom_spec(const Ast &ast) {
std::cerr << "eval_atom_spec\n";
    // atom_specifier <- as_term _ "&" _ atom_specifier / as_term _ "|" _ atom_specifier / as_term
    PyObject* left_spec;
    PyObject* right_spec;
    PyObject* op;
    left_spec = eval_as_term(*ast.nodes[0]);
    switch (ast.choice) {
        case 0:
            op = py_symbols[OP_UNION];
            right_spec = eval_atom_spec(*ast.nodes[1]);
            break;
        case 1:
            op = py_symbols[OP_INTERSECT];
            right_spec = eval_atom_spec(*ast.nodes[1]);
            break;
        case 2:
            op = right_spec = Py_None;
            break;
    }
        
    auto args = PyTuple_New(3);
    if (args == nullptr) {
        throw std::runtime_error("Could not create tuple");
    }
    PyTuple_SET_ITEM(args, 0, op);
    PyTuple_SET_ITEM(args, 1, left_spec);
    PyTuple_SET_ITEM(args, 2, right_spec);

    auto kw_args = PyDict_New();
    if (kw_args == nullptr) {
        Py_DECREF(args);
        throw std::runtime_error("Cannot create keyword dictionary");
    }
    if (PyDict_SetItemString(kw_args, "add_implied", add_implied ? Py_True : Py_False) < 0) {
        Py_DECREF(args);
        Py_DECREF(kw_args);
        throw SemanticsError(use_python_error);
    }
    auto atom_spec = PyObject_Call(AtomSpec_class, args, kw_args);
    if (atom_spec == nullptr) {
        Py_DECREF(args);
        Py_DECREF(kw_args);
        throw SemanticsError(use_python_error);
    }
    return atom_spec;
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
    int c_add_implied;
    if (!PyArg_ParseTuple(args, "OsOOp", &session, &text, &parse_error_class,
            &semantics_error_class, &c_add_implied))
        return nullptr;
std::cerr << "parse text: " << text << "\n";

    add_implied = static_cast<bool>(c_add_implied);
    spec_parser.set_logger([](size_t line, size_t col, const std::string& msg) {
        err_valid = true;
        err_line = line;
        err_col = col;
        err_msg = msg;
    });
    std::shared_ptr<peg::Ast> ast;
    if (spec_parser.parse(text, ast)) {
        //TODO: Check if optimized AST is usable.  I suspect that ::name=="CYS" produces an unusable AST
        // because it skips levels
        print_ast(*ast);
        try {
            return eval_atom_spec(*ast);
        } catch (SemanticsError& e) {
            set_error_info(semantics_error_class, e.what());
        } catch (UserError& e) {
            set_error_info(UserError_class, e.what());
        }
    } else {
        set_error_info(parse_error_class, err_msg);
    }
    return nullptr;
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

#if PERCENT_WHITESPACE
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
#else
// had to not use %whitespace because using the tokenizing operator (<>) to enforce no spaces around '.'
// characters in a model hierarchy stopped the AST from descending below that level
static auto grammar = (R"---(
    atom_specifier <- as_term _ "&" _ atom_specifier / as_term _ "|" _ atom_specifier / as_term
    as_term <- "(" _ atom_specifier _ ")" _ zone_selector? / "~" _ as_term _ zone_selector? / SELECTOR_NAME _ zone_selector? / model_list
    model_list <- model+
    model <- HASH_TYPE _ model_hierarchy (_ "##" _ attribute_list)? _ model_parts* _ zone_selector? / "##" _ attribute_list _ model_parts* _ zone_selector? / model_parts _ zone_selector?
    model_hierarchy <- model_range_list ("." model_range_list)*
    model_range_list <- model_range ("," _ model_range)*
    model_range <- MODEL_SPEC_START _ "-" _ MODEL_SPEC_END / MODEL_SPEC_ANY
    model_parts <- chain+
    chain <- "/" _ part_list (_ "//" _ attribute_list)? _ chain_parts* / "//" _ attribute_list _ chain_parts* / chain_parts+
    chain_parts <- residue+
    residue <- ":" _ part_list ("::" _ attribute_list)? _ residue_parts* / "::" _ attribute_list _ residue_parts* / residue_parts+
    part_list <- PART_RANGE_LIST "," _ part_list / PART_RANGE_LIST
    residue_parts <- atom+
    # atom ranges are not allowed
    atom <- "@" _ atom_list (_ "@@" _ attribute_list)? / "@@" _ attribute_list
    atom_list <- ATOM_NAME "," _ atom_list / ATOM_NAME
    attribute_list <- attr_test ("," _ attr_test)*
    attr_test <- ATTR_NAME _ ATTR_OPERATOR _ ATTR_VALUE / "^" _ ATTR_NAME / ATTR_NAME
    zone_selector <- ZONE_OPERATOR _ real_number / ZONE_OPERATOR _ integer
    ATOM_NAME <- < [-+a-zA-Z0-9_'"*?\[\]\\]+ >
    ATTR_NAME <- < [a-zA-Z_] [a-zA-Z0-9_]* >
    ATTR_OPERATOR <- ">=" | ">" | "<=" | "<" | "==" | "=" | "!==" | "!=" | "<>"
    ATTR_VALUE <- '"' < [^"]+ > '"' / "'" < [^']+ > "'" / [^#/:@,;"' ]+
    HASH_TYPE <- "#!" / "#"
    # limit model numbers to 5 digits to avoid conflicts with hex colors
    MODEL_SPEC <- < [0-9]{1,5} > ![0-9A-Fa-f]
    MODEL_SPEC_ANY <- MODEL_SPEC / "*"
    MODEL_SPEC_END <- MODEL_SPEC / "end" / "*"
    MODEL_SPEC_START <- MODEL_SPEC / "start" / "*"
    RANGE_CHAR <- [A-Za-z0-9_'"*?\[\]\\]
    RANGE_PART <- < "-"? RANGE_CHAR+ >
    PART_RANGE_LIST <- < RANGE_PART _ "-" _ RANGE_PART > / RANGE_PART
    SELECTOR_NAME <- < [a-zA-Z_][-+a-zA-Z0-9_]* >
    ZONE_OPERATOR <- "@>" | "@<" | ":>" | ":<" | "/>" | "/<" | "#>" | "#<"
    EndOfLine <- "\r\n" / "\n" / "\r"
    ~_ <- (' ' / '\t' / EndOfLine)*
    integer <- < [1-9][0-9]* >
    real_number <- < [0-9]* '.' [0-9]+ >
)---");
#endif

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

PyMODINIT_FUNC PyInit__spec_parser()
{
    auto mod = PyModule_Create(&spec_parser_def);
    spec_parser.load_grammar(grammar);
    if (static_cast<bool>(spec_parser) == false) {
        PyErr_SetString(PyExc_SyntaxError, "Atom-specifier grammar is bad");
        return nullptr;
    }
    spec_parser.enable_ast();

    for (auto& sym: symbols)
        py_symbols.push_back(PyUnicode_FromString(sym.c_str()));

    AtomSpec_class = get_module_attribute("chimerax.core.commands.atomspec", "AtomSpec");
    if (AtomSpec_class == nullptr)
        return nullptr;
    ColorArg_class = get_module_attribute("chimerax.core.commands", "ColorArg");
    if (ColorArg_class == nullptr)
        return nullptr;
    TinyArray_class = get_module_attribute("tinyarray", "array");
    if (TinyArray_class == nullptr)
        return nullptr;
    UserError_class = get_module_attribute("chimerax.core.errors", "UserError");
    if (UserError_class == nullptr)
        return nullptr;
    _Atom_class = get_module_attribute("chimerax.core.commands.atomspec", "_Atom");
    if (_Atom_class == nullptr)
        return nullptr;
    _AttrList_class = get_module_attribute("chimerax.core.commands.atomspec", "_AttrList");
    if (_AttrList_class == nullptr)
        return nullptr;
    _AttrTest_class = get_module_attribute("chimerax.core.commands.atomspec", "_AttrTest");
    if (_AttrTest_class == nullptr)
        return nullptr;
    _Chain_class = get_module_attribute("chimerax.core.commands.atomspec", "_Chain");
    if (_Chain_class == nullptr)
        return nullptr;
    _Invert_class = get_module_attribute("chimerax.core.commands.atomspec", "_Invert");
    if (_Invert_class == nullptr)
        return nullptr;
    _Model_class = get_module_attribute("chimerax.core.commands.atomspec", "_Model");
    if (_Model_class == nullptr)
        return nullptr;
    _ModelHierarchy_class = get_module_attribute("chimerax.core.commands.atomspec", "_ModelHierarchy");
    if (_ModelHierarchy_class == nullptr)
        return nullptr;
    _ModelList_class = get_module_attribute("chimerax.core.commands.atomspec", "_ModelList");
    if (_ModelList_class == nullptr)
        return nullptr;
    _ModelRange_class = get_module_attribute("chimerax.core.commands.atomspec", "_ModelRange");
    if (_ModelRange_class == nullptr)
        return nullptr;
    _ModelRangeList_class = get_module_attribute("chimerax.core.commands.atomspec", "_ModelRangeList");
    if (_ModelRangeList_class == nullptr)
        return nullptr;
    _Part_class = get_module_attribute("chimerax.core.commands.atomspec", "_Part");
    if (_Part_class == nullptr)
        return nullptr;
    _PartList_class = get_module_attribute("chimerax.core.commands.atomspec", "_PartList");
    if (_PartList_class == nullptr)
        return nullptr;
    _Residue_class = get_module_attribute("chimerax.core.commands.atomspec", "_Residue");
    if (_Residue_class == nullptr)
        return nullptr;
    _SelectorName_class = get_module_attribute("chimerax.core.commands.atomspec", "_SelectorName");
    if (_SelectorName_class == nullptr)
        return nullptr;
    _Term_class = get_module_attribute("chimerax.core.commands.atomspec", "_Term");
    if (_Term_class == nullptr)
        return nullptr;
    _ZoneSelector_class = get_module_attribute("chimerax.core.commands.atomspec", "_ZoneSelector");
    if (_ZoneSelector_class == nullptr)
        return nullptr;

    add_part_arg = PyUnicode_FromString("add_part");
    if (add_part_arg == nullptr)
        return nullptr;
    add_parts_arg = PyUnicode_FromString("add_parts");
    if (add_parts_arg == nullptr)
        return nullptr;
    append_arg = PyUnicode_FromString("append");
    if (append_arg == nullptr)
        return nullptr;
    uint8x4_arg = PyUnicode_FromString("uint8x4");
    if (uint8x4_arg == nullptr)
        return nullptr;

    auto make_converter_func = get_module_attribute("chimerax.core.commands", "make_converter");
    if (make_converter_func == nullptr)
        return nullptr;
    color_converter_func = PyObject_CallOneArg(make_converter_func, ColorArg_class);
    if (color_converter_func == nullptr)
        return nullptr;
    get_selector_func = get_module_attribute("chimerax.core.commands.atomspec", "get_selector");
    if (get_selector_func == nullptr)
        return nullptr;

    op_eq = get_module_attribute("operator", "eq");
    if (op_eq == nullptr)
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
    op_ne = get_module_attribute("operator", "ne");
    if (op_ne == nullptr)
        return nullptr;
    op_not = get_module_attribute("operator", "not_");
    if (op_not == nullptr)
        return nullptr;
    op_truth = get_module_attribute("operator", "truth");
    if (op_truth == nullptr)
        return nullptr;

    end_string = PyUnicode_FromString("end");
    if (end_string == nullptr)
        return nullptr;
    star_string = PyUnicode_FromString("*");
    if (star_string == nullptr)
        return nullptr;
    start_string = PyUnicode_FromString("start");
    if (start_string == nullptr)
        return nullptr;

    return mod;
}
