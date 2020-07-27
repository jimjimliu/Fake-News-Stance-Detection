import os
import pickle
from csv import DictReader
import pandas as pd


# directory where all generated and saved models at
MODEL_PATH = os.path.join(os.path.join(os.getcwd(), 'model'))
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
DATA_PATH = os.path.join(os.path.join(os.getcwd(), 'data'))
SUBMISSION_PATH = os.path.join(DATA_PATH, 'submissions')

class utils():

    def save_pkl_model(model, file_name, path=MODEL_PATH):
        '''
        save a model using python pickle

        :param model: (Object) the model to be saved
        :param file_name: (String) file name string
        :param path: (String) directory of position to save the model. Default=.../model
        :return: None
        '''
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)

        with open(os.path.join(os.path.join(path, file_name)), 'wb') as handle:
            pickle.dump(model, handle)
        print(file_name, 'saved at: ', os.path.join(os.path.join(os.getcwd(), 'model', file_name)))

    def load_pkl(file_name, path=MODEL_PATH):
        '''
        load an existing model using pickle

        :param file_name: (String) file name
        :param path: (String) directory where model is saved. Default=.../model
        :return: (Object) the model
        '''
        with open(os.path.join(path, file_name), 'rb') as handle:
            model_pkl = pickle.load(handle)

        return model_pkl

    def read(filename):
        rows = []
        with open(os.path.join(DATA_PATH, filename), "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

    def write_out(lst,  predictions, file_name):
        '''
        write answer.csv(submisson file) out

        :param lst: (List) 2-d list
        :param predictions: (List) 1-d list contains prediction labels
        :return: None
        '''

        result = pd.DataFrame(columns=('Headline', 'Body ID', 'Stance'))
        for item in lst:
            result = result.append(
                pd.DataFrame({'Headline': [item['Headline']], 'Body ID': [item['Body ID']], 'Stance': [None]}),
                ignore_index=True)

        for i in range(len(predictions)):
            result.loc[i, 'Stance'] = LABELS[predictions[i]]

        if not os.path.exists(SUBMISSION_PATH):
            os.mkdir(SUBMISSION_PATH)
        result.to_csv(os.path.join(SUBMISSION_PATH, file_name+'.csv'), index=False, encoding='utf-8')  # From pandas library
        print("Submission.csv is saved at: ", os.path.join(SUBMISSION_PATH, file_name+'.csv'))

    def check_balanced(df, column):
        '''
        check if the data set is balanced using the given column as label.
        Return each label's distribution percentage to see if the dataset is balanced.
        :return:
            (Dict): dictionary has key=label_name, value=label percentage
        '''

        # vc is pandas series
        vc = df[column].value_counts()
        # row counts of df
        row_count = df.shape[0]

        result_dict = {}
        for i, v in vc.items():
            result_dict[i] = v, v/row_count

        return result_dict

    def print_info(text=None):
        print("================================================")
        print("|                                              |")
        print("|", text, " "*(len("================================================")-len(text)-5) , "|")
        print("|                                              |")
        print("================================================")



