def test_report_sequences_keeps_integer_keys(ensure_chimerax_initialized):
    # Trac #21065: the query sequence lives under key 0 and hits under their
    # integer ids, so the signal must deliver the dict unchanged.
    from chimerax.blastprotein.ui.results import BlastResultsWorker

    worker = BlastResultsWorker(None, None)
    received = []
    worker.report_sequences.connect(received.append)
    sequences = {0: ("query", "MKV"), 7: ("hit", "MRV")}
    worker.report_sequences.emit(sequences)
    assert received == [sequences]
