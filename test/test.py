import glob
import os

import numpy as np
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import SimpleITK as sitk

data_dir = "C:\\GitHub\\GitHub\\NuclearMedEval\\test\\data\\exampleForTestsData"
result_file = "C:\\GitHub\\GitHub\\NuclearMedEval\\test\\data\\pymiaOutput\\results.csv"
result_summary_file = "C:\\GitHub\\GitHub\\NuclearMedEval\\test\\data\\pymiaOutput\\results_summary.csv"



metrics = [metric.InterclassCorrelation()]


labels = {1: 'WHITEMATTER',
          2: 'GREYMATTER',
          5: 'THALAMUS'
          }
evaluator = eval_.SegmentationEvaluator(metrics, labels)

          # get subjects to evaluate
subject_dirs = [subject for subject in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(subject) and os.path.basename(subject).startswith('Subject')]

for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    print(f'Evaluating {subject_id}...')

    # load ground truth image and create artificial prediction by erosion
    ground_truth = sitk.ReadImage(os.path.join(subject_dir, f'{subject_id}_GT.mha'))
    prediction = ground_truth
    for label_val in labels.keys():
        # erode each label we are going to evaluate
        prediction = sitk.BinaryErode(prediction, [1] * prediction.GetDimension(), sitk.sitkBall, 0, label_val)

    # evaluate the "prediction" against the ground truth
    evaluator.evaluate(prediction, ground_truth, subject_id)

writer.CSVWriter(result_file).write(evaluator.results)

print('\nSubject-wise results...')
writer.ConsoleWriter().write(evaluator.results)

functions = {'MEAN': np.mean, 'STD': np.std}
writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
print('\nAggregated statistic results...')
writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)





