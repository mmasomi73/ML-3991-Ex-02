class OutputWriter:
    path = ''
    data = ''
    algos = ''

    def __init__(self, path, algos, data):
        self.path = path
        self.data = data
        self.algos = algos

    def write(self):
        file_name = self.path + self.algos + '.csv'
        file = open(file_name, "w+")
        file.write(self.algos+' Results:\n')
        file.write('False Alarm Rate,{}\n'.format(self.data['far']))
        file.write('Missing Alarm Rate,{}\n'.format(self.data['mar']))
        file.write('Accuracy Rate,{}\n'.format(self.data['acc']))
        file.write('Train Time,{}\n'.format(self.data['tr']))
        file.write('Test Time,{}\n'.format(self.data['te']))
        file.write('True Positive,{}\n'.format(self.data['tp']))
        file.write('True Negative,{}\n'.format(self.data['tn']))
        file.write('False Positive,{}\n'.format(self.data['fp']))
        file.write('False Negative,{}\n'.format(self.data['fn']))
