import pandas as pd
import mat73
import pickle


def Import_Dict_of_Dataframes(path):
    with open(path, 'rb') as inputFile:
        Data_dataframe = pickle.load(inputFile)
    return Data_dataframe


def load_table_from_struct(path):
    mdata = mat73.loadmat(path)  # To import structure.mat format
    SampleNames = list(mdata['Features'].keys())

    def Extract(lst):
        return [item[0] for item in lst]

    Data, FeatureNames, TimeWindowIndex, Data_dataframe, indexNames = dict(), dict(), dict(), dict(), dict()
    for i in range(len(SampleNames)):
        Data[SampleNames[i]] = mdata['Features'][SampleNames[i]]['Data']
        FeatureNames[SampleNames[i]] = mdata['Features'][SampleNames[i]]['FeatureNames']
        TimeWindowIndex[SampleNames[i]] = mdata['Features'][SampleNames[i]]['TimeWindowIndex']
        indexNames = Extract(TimeWindowIndex[SampleNames[i]])
        Data_dataframe[SampleNames[i]] = pd.DataFrame(Data[SampleNames[i]], index=indexNames,
                                                      columns=FeatureNames[SampleNames[i]])
    del (Data, FeatureNames, TimeWindowIndex, indexNames, mdata, SampleNames)

    return Data_dataframe


#################################################################################################################
if __name__ == '__main__':
    LW_Int = pd.DataFrame([[5000, 5000],
                           [5000, 1000],
                           [5000, 500],  # 3(larger than 2 GB use MAT - file version 7.3 or later)
                           [5000, 100],  # 4(larger than 2 GB use MAT - file version 7.3 or later)
                           [1000, 1000],
                           [1000, 500],
                           [1000, 100],  # 7(larger than 2 GB use MAT - file version 7.3 or later)
                           [500, 500],
                           [500, 100],
                           [100, 100]], columns=['LW', 'Int'])
    LW = LW_Int['LW'][7]  # 500
    Int = LW_Int['Int'][7]  # 500
    name = 'Features_LW' + str(LW) + 'Int' + str(Int) + 'Cycle'
    dir_path = 'data/raw_data/'
    print('Load Dataset ' + str(name) + '.mat ...')
    path = dir_path + name + '.mat'
    Data_dataframe = load_table_from_struct(path)
    print(Data_dataframe)
