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

static parser spec_parser;
static PyObject* session;
static PyObject *parse_error_class, *semantics_error_class;
static PyObject* AtomSpec_class;
static PyObject* _Model_class;
static PyObject* _ModelHierarchy_class;
static PyObject* _ModelList_class;
static PyObject* _ModelRange_class;
static PyObject* _ModelRangeList_class;
static PyObject* _Term_class;
static PyObject* append_arg;
static std::string use_python_error("Use Python error");
static bool add_implied, order_implicit_atoms, outermost_inversion;

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

// Fixed text strings that need to be handed off to the Python layer
enum Symbols { OP_UNION, OP_INTERSECT, NUM_SYMBOLS };
static std::vector<std::string> symbols = { "|", "&" };
static std::vector<PyObject*> py_symbols;

void
print_ast(const Ast & ast)
{
    std::cerr << ast.name << " '" << ast.token_to_string() << "'; choice " << ast.choice << "  " << ast.nodes.size() << " subnodes\n";
    for (auto node: ast.nodes) {
        print_ast(*node);
    }
}
    
// NOTE: to figure out how things work in the Python layer you have to not only look at the
// corresponding parsing class, but also the corresponding method of _AtomSpecSemantics

static PyObject*
eval_zone_selector(const Ast &ast) {
std::cerr << "eval_zone_selector\n";
    // zone_selector <- ZONE_OPERATOR _ real_number / ZONE_OPERATOR _ integer
    //TODO
    return Py_None;
}

static PyObject*
eval_model_parts(const Ast &ast) {
std::cerr << "eval_model_parts\n";
    // model_parts <- chain+
    //TODO
    return Py_None;
}

static PyObject*
eval_attribute_list(const Ast &ast) {
std::cerr << "eval_attribute_list\n";
    // attribute_list <- attr_test ("," _ attr_test)*
    //TODO
    return Py_None;
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
    //TODO
    return Py_None;
}

static PyObject*
eval_MODEL_SPEC_END(const Ast &ast) {
std::cerr << "eval_MODEL_SPEC_END\n";
    // MODEL_SPEC_END <- MODEL_SPEC / "end" / "*"
    //TODO
    return Py_None;
}

static PyObject*
eval_MODEL_SPEC_ANY(const Ast &ast) {
std::cerr << "eval_MODEL_SPEC_ANY\n";
    // MODEL_SPEC_ANY <- MODEL_SPEC / "*"
    if (ast.choice == 0)
        return eval_MODEL_SPEC(*ast.nodes[0]);
    auto text = PyUnicode_FromString(ast.token_to_string().c_str());
    if (text == nullptr)
        return nullptr;
    return text;
}

static PyObject*
eval_model_range(const Ast &ast) {
std::cerr << "eval_model_range\n";
    // model_range <- MODEL_SPEC_START _ "-" _ MODEL_SPEC_END / MODEL_SPEC_ANY
    PyObject* left;
    PyObject* right;
    if (ast.choice == 0) {
        left = eval_MODEL_SPEC_START(*ast.nodes[0]);
        right = eval_MODEL_SPEC_START(*ast.nodes[1]);
    } else {
        left = eval_MODEL_SPEC_ANY(*ast.nodes[0]);
        right = Py_None;
    }
    auto mr = PyObject_CallFunctionObjArgs(_ModelRange_class, left, right, nullptr);
    if (mr == nullptr) {
        set_error_info(semantics_error_class, use_python_error);
        return nullptr;
    }
    return mr;
}

static PyObject*
eval_model_range_list(const Ast &ast) {
std::cerr << "eval_model_range_list\n";
    // model_range_list <- model_range ("," _ model_range)*
    auto range = eval_model_range(*ast.nodes[0]);
    if (range == nullptr)
        return nullptr;
    auto mrl = PyObject_CallFunctionObjArgs(_ModelRangeList_class, range, nullptr);
    if (mrl == nullptr) {
        set_error_info(semantics_error_class, use_python_error);
        return nullptr;
    }
    for (auto i = 1u; i < ast.nodes.size(); ++i) {
        range = eval_model_range(*ast.nodes[i]);
        if (range == nullptr) {
            set_error_info(semantics_error_class, use_python_error);
            return nullptr;
        }
        if (PyObject_CallMethodOneArg(mrl, append_arg, range) == nullptr) {
            Py_DECREF(mrl);
            throw std::logic_error(use_python_error);
        }
    }
    return mrl;
}

static PyObject*
eval_model_hierarchy(const Ast &ast) {
std::cerr << "eval_model_hierarchy\n";
    // model_hierarchy <- model_range_list ("." model_range_list)*
    auto rl = eval_model_range_list(*ast.nodes[0]);
    if (rl == nullptr)
        return nullptr;
    auto hierarchy = PyObject_CallFunctionObjArgs(_ModelHierarchy_class, rl, nullptr);
    if (hierarchy == nullptr) {
        set_error_info(semantics_error_class, use_python_error);
        return nullptr;
    }
    for (auto i = 1u; i < ast.nodes.size(); ++i) {
        rl = eval_model_range_list(*ast.nodes[i]);
        if (rl == nullptr) {
            set_error_info(semantics_error_class, use_python_error);
            return nullptr;
        }
        if (PyObject_CallMethodOneArg(hierarchy, append_arg, rl) == nullptr) {
            Py_DECREF(hierarchy);
            throw std::logic_error(use_python_error);
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
    PyObject* parts = Py_None;
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
    if (model == nullptr) {
        set_error_info(semantics_error_class, use_python_error);
        return nullptr;
    }
    if (parts != Py_None) {
        //TODO
    }
    if (zone == Py_None)
        return model;
    
    if (PyObject_SetAttrString(zone, "model", model) < 0) {
        Py_DECREF(model);
        set_error_info(semantics_error_class, use_python_error);
        return nullptr;
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
            set_error_info(semantics_error_class, use_python_error);
            return nullptr;
        }
        if (PyObject_CallMethodOneArg(py_ml, append_arg, model) == nullptr) {
            Py_DECREF(py_ml);
            throw std::logic_error(use_python_error);
        }
    }
    return py_ml;
}

static PyObject*
eval_as_term(const Ast &ast) {
std::cerr << "eval_as_term\n";
    // as_term <- "(" _ atom_specifier _ ")" _ zone_selector? / "~" _ as_term _ zone_selector? / SELECTOR_NAME _ zone_selector? / model_list
    switch (ast.choice) {
        case 0:
        case 1:
        case 2:
            //TODO
            return Py_None;
    }
    return PyObject_CallOneArg(_Term_class, eval_model_list(*ast.nodes[0]));
}
    
static PyObject*
eval_atom_spec(const Ast &ast) {
std::cerr << "eval_atom_spec\n";
    print_ast(ast);
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
        throw std::logic_error(use_python_error);
    }
    return PyObject_Call(AtomSpec_class, args, kw_args);
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
        return eval_atom_spec(*ast);
    }
    set_error_info(parse_error_class, err_msg);
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
    _Term_class = get_module_attribute("chimerax.core.commands.atomspec", "_Term");
    if (_Term_class == nullptr)
        return nullptr;

    append_arg = PyUnicode_FromString("append");
    if (append_arg == nullptr)
        return nullptr;

    return mod;
}
