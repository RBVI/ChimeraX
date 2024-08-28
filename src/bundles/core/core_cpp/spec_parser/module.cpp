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

#include <vector>
#include <utility>

#include "Python.h"
#include "peglib.h"

//#include <logger/logger.h>

using namespace peg;

static parser spec_parser;
static PyObject* objects_class;
static PyObject* objects_intersect;
static PyObject* objects_union;
static PyObject* get_selector;
static PyObject* add_implied_bonds;
static PyObject* session;
static PyObject* models;
static PyObject* combine_arg;
static PyObject* invert_arg;
static PyObject* list_arg;
static PyObject* add_model_arg;
static PyObject* add_atoms_arg;
static PyObject* atomspec_has_atoms_arg;
static std::string use_python_error("Use Python error");
static bool add_implied, order_implicit_atoms, outermost_inversion;

static const char*
docstr_evaluate = \
"evaluate(session, text)\n" \
"\n" \
"Evaluate the given text for an initial atom spec and if one is found" \
" return a tuple containing a chimerax.core.objects.Objects instance," \
" the part of text used for the atom spec, and the remaining text.";

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
    GlobalModelMatcher(bool em, std::vector<std::vector<ModelMatcher>> ls) { exact_match = em; levels = ls; }
    std::vector<PyObject*>  matches();
};

std::vector<PyObject*>
GlobalModelMatcher::matches()
{
    std::vector<PyObject*> matched_models;
    auto list_size = PySequence_Fast_GET_SIZE(models);
    for (decltype(list_size) i = 0; i < list_size; ++i) {
        auto m = PySequence_Fast_GET_ITEM(models, i);
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

static PyObject*
new_models_objects_instance(std::vector<PyObject*> init_models)
{
    auto objects_inst = new_objects_instance();
    for (auto py_model: init_models) {
        if (PyObject_CallMethodOneArg(objects_inst, add_model_arg, py_model) == nullptr) {
            Py_DECREF(objects_inst);
            throw std::logic_error(use_python_error);
        }
        // kludge so that for now some things gets selected, needs work once sub-parts are supported
        auto ret = PyObject_CallMethodNoArgs(py_model, atomspec_has_atoms_arg);
        if (!PyBool_Check(ret)) {
            Py_DECREF(objects_inst);
            Py_DECREF(ret);
            throw std::logic_error("model.atomspec_has_atom() did not return a boolean value");
        }
        if (ret == Py_True) {
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
                Py_DECREF(no_args);
                throw std::runtime_error("Cannot create keyword dictionary");
            }
            if (PyDict_SetItemString(kw_args, "ordered", order_implicit_atoms ? Py_True : Py_False) < 0) {
                Py_DECREF(no_args);
                Py_DECREF(kw_args);
                Py_DECREF(objects_inst);
                throw std::logic_error(use_python_error);
            }
            auto py_atoms = PyObject_Call(method, no_args, kw_args);
            if (PyObject_CallMethodOneArg(objects_inst, add_atoms_arg, py_atoms) == nullptr) {
                Py_DECREF(objects_inst);
                Py_DECREF(py_atoms);
                Py_DECREF(no_args);
                Py_DECREF(kw_args);
                throw std::logic_error(use_python_error);
            }
            Py_DECREF(no_args);
            Py_DECREF(kw_args);
            Py_DECREF(py_atoms);
        }
        Py_DECREF(ret);
    }
    return objects_inst;
}

static size_t err_line, err_col;
static std::string err_msg;
static bool err_valid;

static void
set_error_info(PyObject* err_type, std::string msg)
{
    PyObject *type, *value, *traceback;
    auto err_val = PyTuple_New(2);
    if (err_val == nullptr)
        throw std::runtime_error("Could not create tuple");
    PyErr_Fetch(&type, &value, &traceback);
    PyTuple_SetItem(err_val, 0, PyLong_FromSize_t( err_valid ? err_col-1 : (size_t)0));
    PyTuple_SetItem(err_val, 1, (msg == use_python_error ? value : PyUnicode_FromString(msg.c_str())));
    PyErr_Restore(err_type, err_val, traceback);
}

//TODO: accept find_match() keywords
extern "C" PyObject *
evaluate(PyObject *, PyObject *args)
{
    const char* text;
    int c_quoted, c_add_implied, c_order_implicit;
    PyObject *parse_error_class, *semantics_error_class;
    if (!PyArg_ParseTuple(args, "OOspOOpp", &session, &models, &text, &c_quoted, &parse_error_class,
            &semantics_error_class, &c_add_implied, &c_order_implicit))
        return nullptr;
    bool quoted = static_cast<bool>(c_quoted);
    add_implied = static_cast<bool>(c_add_implied);
    order_implicit_atoms = static_cast<bool>(c_order_implicit);
    models = PySequence_Fast(models, "evaluate() arg 'models' is not a sequence");
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
std::cerr << "Parsing text " << text << "\n";
        err_valid = false;
        spec_parser.parse(text, returned_objects_instance);
        if (returned_objects_instance == nullptr && !quoted) {
            // progressively lop off spaced-separated text and see if what remain is legal...
            auto space_pos = trial_text.find_last_of(' ');
            while (space_pos != std::string::npos) {
                trial_text = trial_text.substr(0, space_pos);
std::cerr << "Parsing text " << trial_text << "\n";
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
    } catch (std::logic_error& e) {
        Py_DECREF(models);
        try {
            set_error_info(semantics_error_class, e.what());
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
    atom_specifier <- as_term "&" atom_specifier / as_term "|" atom_specifier / as_term
    as_term <- "(" atom_specifier ")" zone_selector? / "~" as_term zone_selector? / SELECTOR_NAME zone_selector? / model_list
    model_list <- model+
    model <- HASH_TYPE model_hierarchy zone_selector?
    model_hierarchy <- < model_range_list (!Space "." !Space model_hierarchy)* >
    model_range_list <- model_range ("," model_range_list)*
    model_range <- MODEL_SPEC_START "-" MODEL_SPEC_END / MODEL_SPEC_ANY
    zone_selector <- ZONE_OPERATOR real_number / ZONE_OPERATOR integer
    # limit model numbers to 5 digits to avoid conflicts with hex colors
    HASH_TYPE <- "#!" / "#"
    MODEL_SPEC <- < [0-9]{1,5} > ![0-9A-Fa-f]
    MODEL_SPEC_ANY <- MODEL_SPEC / "*"
    MODEL_SPEC_END <- MODEL_SPEC / "end" / "*"
    MODEL_SPEC_START <- MODEL_SPEC / "start" / "*"
    SELECTOR_NAME <- < [a-zA-Z_][-+a-zA-Z0-9_]* >
    ZONE_OPERATOR <- "@>" | "@<" | ":>" | ":<" | "/>" | "/<" | "#>" | "#<"
    EndOfLine <- "\r\n" / "\n" / "\r"
    Space <- ' ' / '\t' / EndOfLine
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
    atomspec_has_atoms_arg = PyUnicode_FromString("atomspec_has_atoms");
    if (atomspec_has_atoms_arg == nullptr)
        return nullptr;

    // atom_specifier
    spec_parser["atom_specifier"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " atom_specifier semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        std::cerr << "choice: " << vs.choice() << "\n";
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
        std::cerr << vs.size() << " as_term semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        std::cerr << "choice: " << vs.choice() << "\n";
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
            PyTuple_SetItem(args, 1, models);
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
            
    // model_list
    spec_parser["model_list"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " model_list semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
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
            
    // model
    spec_parser["model"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " model semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        auto gmatcher = GlobalModelMatcher(std::any_cast<bool>(vs[0]), 
            std::any_cast<std::vector<std::vector<ModelMatcher>>>(vs[1]));
        auto objects = new_models_objects_instance(gmatcher.matches());
        //TODO: zone
        return objects;
    };
            
    // model_hierarchy
    spec_parser["model_hierarchy"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " model_hierarchy semantic values\n";
        std::cerr << "model_hierarchy choice:" << vs.choice() << "\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
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
        std::cerr << vs.size() << " model_range_list semantic values\n";
        std::cerr << "model_range_list choice:" << vs.choice() << "\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
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
        std::cerr << vs.size() << " model_range semantic values\n";
        std::cerr << "model_range choice:" << vs.choice() << "\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        if (vs.choice() == 0)
            return ModelMatcher(std::any_cast<ModelPart>(vs[0]), std::any_cast<ModelPart>(vs[1]));
        return ModelMatcher(std::any_cast<ModelPart>(vs[0]));
    };
            
    // HASH_TYPE
    spec_parser["HASH_TYPE"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " HASH_TYPE semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        return vs.tokens[0][1] == '!';
    };
            
    // MODEL_SPEC
    spec_parser["MODEL_SPEC"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " MODEL_SPEC semantic values\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        return ModelPart(vs.token_to_number<int>());
    };
            
    // MODEL_SPEC_ANY
    spec_parser["MODEL_SPEC_ANY"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " MODEL_SPEC_ANY semantic values\n";
        std::cerr << "MODEL_SPEC_ANY choice:" << vs.choice() << "\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        if (vs.choice() == 1)
            return ModelPart();
        return std::any_cast<ModelPart>(vs[0]);
    };
            
    // MODEL_SPEC_START
    spec_parser["MODEL_SPEC_START"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " MODEL_SPEC_START semantic values\n";
        std::cerr << "MODEL_SPEC_START choice:" << vs.choice() << "\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        if (vs.choice() > 0)
            return ModelPart();
        return std::any_cast<ModelPart>(vs[0]);
    };
            
    // MODEL_SPEC_END
    spec_parser["MODEL_SPEC_END"] = [](const SemanticValues &vs) {
        std::cerr << vs.size() << " MODEL_SPEC_END semantic values\n";
        std::cerr << "MODEL_SPEC_END choice:" << vs.choice() << "\n";
        std::cerr << "tokens:";
        for (auto tk: vs.tokens)
            std::cerr << " " << tk;
        std::cerr << "\n";
        if (vs.choice() > 0)
            return ModelPart();
        return std::any_cast<ModelPart>(vs[0]);
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
        return vs.token_to_number<int>();
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
