# data/

Drop one Stooq-format daily CSV per ticker here:

    Date,Open,High,Low,Close,Volume
    2015-01-02,111.39,111.44,107.35,109.33,53204626

Filenames may be `aapl.us.txt`, `AAPL.csv`, etc. The exchange suffix is stripped
to form the ticker.

This directory is intentionally empty. No prices were generated to stand in for
real data -- see ../FINDINGS.md for why, and for what the code has and has not
established without them.
