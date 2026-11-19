from pathlib import Path
import shutil

from importlib.util import module_from_spec, spec_from_file_location


path = Path(__file__).resolve().parents[1] / "utils" / "find_qt_warmup_symbols.py"
spec = spec_from_file_location("find_qt_warmup_symbols", path)
tool = module_from_spec(spec)
spec.loader.exec_module(tool)


def test_matches_exact_method_and_reports_distinct_exports():
    method = "QObject::moveToThread(QThread*)"
    exports = [
        ("void_export", "public: void __cdecl QObject::moveToThread(class QThread * __ptr64)"),
        ("overload", "QObject::moveToThread(QThread*, int)"),
        ("namespace", "Example::QObject::moveToThread(QThread*)"),
        ("itanium_export", method),
        ("second_export", "public: bool __cdecl QObject::moveToThread(QThread *)"),
        ("itanium_export", method),
    ]
    # MSVC's class annotation is part of its undecorated output.
    matches = tool.find_matches(exports, method)
    assert [symbol for symbol, _ in matches] == ["itanium_export", "second_export", "void_export"]


def test_windows_dumpbin_demangles_only_relevant_exports(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda tool: tool)
    symbol = "?moveToThread@QObject@@QEAA_NPEAVQThread@@@Z"

    def run(*args, **kwargs):
        if args[0] == "dumpbin":
            return f"  1 A 1234 {symbol}\n  2 B 1238 ?ignored@QObject@@123\n"
        assert args == ("undname", symbol)
        return f'Undecoration of :- "{symbol}"\nis :- "public: bool __cdecl QObject::moveToThread(class QThread *)"\n'

    monkeypatch.setattr(tool, "run", run)
    method = "QObject::moveToThread(QThread*)"
    exports = tool.windows_exports(Path("Qt6Core.dll"), [method])
    assert tool.find_matches(exports, method) == [
        (symbol, "public: bool __cdecl QObject::moveToThread(class QThread *)")
    ]


def test_unix_normalizes_export_names_before_demangling(monkeypatch):
    symbol = "_ZNK12QFontMetrics6heightEv"
    signature = "QFontMetrics::height() const"

    def run(*args, input=None):
        if args[0] == "nm":
            if "-arch" in args:
                return f"_{symbol}\n"
            return f"{symbol}@@Qt_6 T 100 20\n"
        assert args == ("c++filt",)
        assert input == symbol + "\n"
        return signature + "\n"

    monkeypatch.setattr(tool, "run", run)
    assert list(tool.unix_exports(Path("QtGui"), "darwin", "arm64")) == [(symbol, signature)]
    assert list(tool.unix_exports(Path("QtGui"), "linux", None)) == [(symbol, signature)]
