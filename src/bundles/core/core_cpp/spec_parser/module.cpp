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

PyMODINIT_FUNC PyInit__spec_parser()
{
    auto mod = PyModule_Create(&spec_parser_def);
    spec_parser.load_grammar(grammar);
    if (static_cast<bool>(spec_parser) == false) {
        PyErr_SetString(PyExc_SyntaxError, "Atom-specifier grammar is bad");
        return nullptr;
    }
    spec_parser.enable_ast();

    return mod;
}
