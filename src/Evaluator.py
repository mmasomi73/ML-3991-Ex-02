

class Evaluator:
    true = []
    metric = 'nab'
    prediction = []

    def __init__(self, true, prediction, metric='nab', numenta_time=None):
        self.true = true
        self.metric = metric
        self.prediction = prediction

    def getConfusionMetrics(self):
        def single_binary(true, prediction):
            true_ = true == 1
            prediction_ = prediction == 1
            TP = (true_ & prediction_).sum()
            TN = (~true_ & ~prediction_).sum()
            FP = (~true_ & prediction_).sum()
            FN = (true_ & ~prediction_).sum()
            return TP, TN, FP, FN

        if type(self.true) != type(list()):
            TP, TN, FP, FN = single_binary(self.true, self.prediction)
        else:
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in range(len(self.true)):
                TP_, TN_, FP_, FN_ = single_binary(self.true[i], self.prediction[i])
                TP, TN, FP, FN = TP + TP_, TN + TN_, FP + FP_, FN + FN_
        return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
