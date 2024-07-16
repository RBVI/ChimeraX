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

#if 0
#include <algorithm>  // for std::sort, std::find
#include <cctype>
#include <cmath> // abs
#include <fstream>
#include <set>
#include <sstream>
#include <stdio.h>  // fgets
#include <unordered_map>
#endif
#include <utility>

#include "Python.h"
#include "peglib.h"

//#include <logger/logger.h>

using namespace peg;

static parser spec_parser;
static PyObject* objects_class;
static PyObject* get_selector;
static PyObject* add_implied_bonds;
static PyObject* session;
static PyObject* combine_arg;
static PyObject* list_arg;
static std::string use_python_error("Use Python error");

static const char*
docstr_evaluate = \
"evaluate(session, text)\n" \
"\n" \
"Evaluate the given text for an initial atom spec and if one is found" \
" return a tuple containing a chimerax.core.objects.Objects instance," \
" the part of text used for the atom spec, and the remaining text.";

static void change_exception_type(PyObject* etype)
{
    //TODO: when we're at Python 3.12, instead use: PyException_SetCause(PyErr_GetRaisedException(), etype);
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_Restore(etype, value, traceback);
}

static PyObject*
new_objects_instance()
{
    auto objects_inst = PyObject_CallNoArgs(objects_class);
    if (objects_inst == nullptr)
        throw std::runtime_error("Cannot create Objects instance");
    return objects_inst;
}

//TODO: accept find_match() keywords
extern "C" PyObject *
evaluate(PyObject *, PyObject *args)
{
    const char* text;
    int quoted;    
    if (!PyArg_ParseTuple(args, "Osp", &session, &text, &quoted))
        return nullptr;
    PyObject* returned_objects_instance = nullptr;
    std::string trial_text = text;
    try {
std::cerr << "Parsing text " << text << "\n";
        spec_parser.parse(text, returned_objects_instance);
        if (returned_objects_instance == nullptr && !quoted) {
            // progressively lop off spaced-separated text and see if what remain is legal...
            auto space_pos = trial_text.find_last_of(' ');
            while (space_pos != std::string::npos) {
                trial_text = trial_text.substr(0, space_pos);
std::cerr << "Parsing text " << trial_text << "\n";
                spec_parser.parse(trial_text.c_str(), returned_objects_instance);
                if (returned_objects_instance != nullptr)
                    break;
                space_pos = trial_text.find_first_of(' ');
            }
        }
    } catch (std::runtime_error& e) {
        if (use_python_error == e.what())
            change_exception_type(PyExc_RuntimeError);
        else
            PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (std::logic_error& e) {
        if (use_python_error == e.what())
            change_exception_type(PyExc_ValueError);
        else
            PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    if (returned_objects_instance == nullptr) {
        PyErr_SetString(PyExc_AssertionError, "parser did not set Objects instance");
        return nullptr;
    }
    auto ret = PyObject_CallOneArg(add_implied_bonds, returned_objects_instance);
    if (ret == nullptr) {
        Py_DECREF(ret);
        throw std::runtime_error(use_python_error);
    }
    auto ret_val = PyTuple_New(3);
    PyTuple_SetItem(ret_val, 0, returned_objects_instance);
    PyTuple_SetItem(ret_val, 1, PyUnicode_FromString(trial_text.c_str()));
    PyTuple_SetItem(ret_val, 2, PyUnicode_FromString(std::string(text).substr(trial_text.size()).c_str()));
    return ret_val;
}

static struct PyMethodDef spec_parser_functions[] =
{
    { "evaluate", (PyCFunction)evaluate, METH_VARARGS, docstr_evaluate },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef spec_parser_def =
{
    PyModuleDef_HEAD_INIT,
    "spec_parser",
    "Parse/evaluate atom specifiers",
    -1,
    spec_parser_functions,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

static auto grammar = (R"---(
    atom_specifier <- as_term
    as_term <- SELECTOR_NAME zone_selector?
    zone_selector <- ZONE_OPERATOR real_number / ZONE_OPERATOR integer
    SELECTOR_NAME <- < [a-zA-Z_][-+a-zA-Z0-9_]* >
    ZONE_OPERATOR <- "@>" | "@<" | ":>" | ":<" | "/>" | "/<" | "#>" | "#<"
    integer <- < [1-9][0-9]* >
    real_number <- < [0-9]* '.' [0-9]+ >
    %whitespace  <-  [ \t\r\n]*
)---");

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
    spec_parser.enable_packrat_parsing();
    objects_class = get_module_attribute("chimerax.core.objects", "Objects");
    if (objects_class == nullptr)
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
    list_arg = PyUnicode_FromString("list");
    if (list_arg == nullptr)
        return nullptr;

    // SELECTOR_NAME
    spec_parser["SELECTOR_NAME"] = [](const SemanticValues &vs) {
        auto sel_text = PyUnicode_FromString(vs.token_to_string().c_str());
        if (sel_text == nullptr)
            throw std::runtime_error("Could not convert token to string");
        auto selector = PyObject_CallOneArg(get_selector, sel_text);
        if (selector == nullptr) {
            Py_DECREF(sel_text);
            throw std::runtime_error("Could not convert token to string");
        }
        Py_DECREF(sel_text);
        if (selector == Py_None) {
            Py_DECREF(selector);
            std::string err_msg = "\"";
            err_msg += vs.token_to_string();
            err_msg += "\" is not a selector name";
            throw std::logic_error(err_msg);
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
                throw std::logic_error(use_python_error);
            }
            Py_DECREF(selector);
        } else {
            auto args = PyTuple_New(3);
            if (args == nullptr) {
                Py_DECREF(selector);
                throw std::runtime_error("Could not create 3-tuple for selector args");
            }
            PyTuple_SetItem(args, 0, session);
            //TODO: actual 'models' arg in selector call
            PyTuple_SetItem(args, 1, PyObject_GetAttrString(session, "models"));
            PyTuple_SetItem(args, 2, selector_objects);
            auto ret = PyObject_CallObject(selector, args);
            Py_DECREF(selector);
            Py_DECREF(args);
            if (ret == nullptr)  
                throw std::logic_error(use_python_error);
            Py_DECREF(ret);
        }
        return selector_objects;
    };
            
    // as_term
    spec_parser["as_term"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " as_term semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        if (vs.size() == 1)
            return std::any_cast<PyObject*>(vs[0]);
        // there's a zone selector
        auto zone_info = std::any_cast<std::pair<std::string_view, float>>(vs[1]);
        auto zone_objects = new_objects_instance();
        auto base_objects = std::any_cast<PyObject*>(vs[0]);
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
            auto models = PyObject_GetAttrString(session, "models");
            auto model_list = PyObject_CallMethodNoArgs(models, list_arg);
            if (model_list == nullptr) {
                Py_DECREF(base_objects);
                Py_DECREF(zone_objects);
                Py_DECREF(model_list);
                throw std::logic_error(use_python_error);
            }
            if (!PyList_Check(model_list)) {
                Py_DECREF(base_objects);
                Py_DECREF(zone_objects);
                Py_DECREF(model_list);
                throw std::logic_error("session.models is not a list!");
            }
            auto list_size = PyList_GET_SIZE(model_list);
            for (decltype(list_size) i = 0; i < list_size; ++i) {
                auto m = PyList_GET_ITEM(model_list, i);
                auto ret = PyObject_CallMethod(m, "atomspec_zone", "OOfCCO", session, coords,
                    zone_info.second, zone_info.first[0], zone_info.first[1], zone_objects);
                if (ret == nullptr) {
                    Py_DECREF(base_objects);
                    Py_DECREF(zone_objects);
                    Py_DECREF(model_list);
                    throw std::logic_error(use_python_error);
                }
            }
            Py_DECREF(model_list);
        }
        Py_DECREF(base_objects);
        return zone_objects;
    };

    // zone_selector
    spec_parser["zone_selector"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " zone_selector semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        return std::make_pair(std::any_cast<std::string_view>(vs[0]), std::any_cast<float>(vs[1]));
    };

    // ZONE_OPERATOR
    spec_parser["ZONE_OPERATOR"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " ZONE_OPERATOR semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        return vs.token();
    };

    // integer
    spec_parser["integer"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " integer semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        return vs.token_to_number<float>();
    };

    // real_number
    spec_parser["real_number"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " real_number semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        return vs.token_to_number<float>();
    };

    return mod;
}
