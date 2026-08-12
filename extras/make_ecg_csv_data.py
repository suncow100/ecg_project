import wfdb
import numpy as np
import pandas as pd

record = wfdb.rdrecord("/home/qortjsdn/projects/ecg_project/mit-bih-arrhythmia-database-1.0.0/233")
sig = record.p_signal[:, 0]
segment = sig[:360 * 30]

pd.Series(segment).to_csv(
    "/mnt/c/Users/백선우/OneDrive/바탕 화면/test_ecg.csv",
    index=False, header=False
)