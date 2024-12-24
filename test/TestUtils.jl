
module TestUtils
using Conda
using PyCall
using Pkg

Conda.pip_interop(true)
Conda.pip("install", "SimpleITK")
Conda.pip("install", "pandas")
Conda.pip("install", "pymia")

sitk = pyimport("SimpleITK")
pym = pyimport("pymia")
np= pyimport("numpy")


####### getting example data

## examples taken from https://pymia.readthedocs.io/en/latest/examples.evaluation.basic.html?highlight=dice#Evaluation-of-results

##### downloading data   
py"""
import argparse
import io
import os
import urllib.request as request
import zipfile

def main(url: str, data_dir: str):
    print(f'Downloading... ({url})')   
    resp = request.urlopen(url)
    zip_ = zipfile.ZipFile(io.BytesIO(resp.read()))
    print(f'Extracting... (to {data_dir})')
    members = zip_.infolist()
    for member in members:
        if member.filename.startswith('Subject_') or member.filename.endswith('.h5'):
            if not os.path.basename(member.filename):
                continue
            zip_.extract(member, data_dir)
"""
py"main"("https://github.com/rundherum/pymia-example-data/releases/download/v0.1.0/example-data.zip", "./test/data/exampleForTestsData")


py"""
import glob
import os

import numpy as np
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import SimpleITK as sitk
data_dir = './test/data/exampleForTestsData'
result_file = './test/data/pymiaOutput/results.csv'
result_summary_file = './test/data/pymiaOutput/results_summary.csv'

def downloadd(url: str, data_dir: str):
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95'), metric.VolumeSimilarity()]
    labels = {1: 'WHITEMATTER',2: 'GREYMATTER'  }
    # get subjects to evaluate
    subject_dirs = [subject for subject in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(subject) and os.path.basename(subject).startswith('Subject')]
    return subject_dirs
"""  

    for subject_dir in subject_dirs:

        subject_id = os.path.basename(subject_dir);ground_truth = sitk.ReadImage(os.path.join(subject_dir, f'{subject_id}_GT.mha'))

        prediction = ground_truth

        for label_val in labels.keys():
            # erode each label we are going to evaluate
            prediction = sitk.BinaryErode(prediction, [1,1,1], sitk.sitkBall, 0, label_val)
"""



    py"main"()

    # print('\nSubject-wise results...')
    # writer.ConsoleWriter().write(evaluator.results)
end
"""

data_dir = "./test/data/exampleForTestsData"
result_file = "./test/data/pymiaOutput/results.csv"
result_summary_file = "./test/data/pymiaOutput/results_summary.csv"

end # module TestUtils
end
end
# Pkg.build("HDF5")
# using HDF5