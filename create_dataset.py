from lib import *
import sys
import pickle
import os

def main(path):
    train=False
    test=False
    validation=False
    
    if '-h' in sys.argv:
        print('help')
        print('Run with: -t [train/test/validation]')
    
    if '-t' and 'train' in sys.argv:
        train=True
        dataName='train'

    if '-t' and 'test' in sys.argv:
        test=True
        dataName='test'
        
    if '-t' and 'validation' in sys.argv:
        validation=True
        dataName='validation'
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(path,'created')
    else:
        print(path,'already there')

    if train:
        data = load_flood_train_data('/nfs/kloe/einstein4/martino/WQeMS/v1.1/data/flood_events/HandLabeled/S1Hand/', '/nfs/kloe/einstein4/martino/WQeMS/v1.1/data/flood_events/HandLabeled/LabelHand/')
        dataset=list()
        for t in np.arange(len(data)):
            dataset.append(processAndAugment(data[t]))
            
        data_list,y_list = get_timeseries_train(dataset)
        selected_data, selected_y = create_dataset(data_list,y_list)
        
        print('saving in',path)
        with open(path+dataName+'_data', "wb") as s_data:   #Pickling
            pickle.dump(selected_data, s_data)

        with open(path+dataName+'_y', "wb") as s_y:   #Pickling
            pickle.dump(selected_y, s_y)
            
    if test:
        data = load_flood_test_data('/nfs/kloe/einstein4/martino/WQeMS/v1.1/data/flood_events/HandLabeled/S1Hand/', '/nfs/kloe/einstein4/martino/WQeMS/v1.1/data/flood_events/HandLabeled/LabelHand/')
        dataset=list()
        for t in np.arange(len(data)):
            dataset.append(processTestIm(data[t]))
        
        data_list,y_list = get_timeseries_test(dataset)
        selected_data, selected_y = create_dataset(data_list,y_list)
        
        print('saving in',path)
        with open(path+dataName+'_data', "wb") as s_data:   #Pickling
            pickle.dump(selected_data, s_data)

        with open(path+dataName+'_y', "wb") as s_y:   #Pickling
            pickle.dump(selected_y, s_y)
            
            
    if validation:
        data = load_flood_valid_data('/nfs/kloe/einstein4/martino/WQeMS/v1.1/data/flood_events/HandLabeled/S1Hand/', '/nfs/kloe/einstein4/martino/WQeMS/v1.1/data/flood_events/HandLabeled/LabelHand/')
        dataset=list()
        for t in np.arange(len(data)):
            dataset.append(processTestIm(data[t]))
            
        data_list,y_list = get_timeseries_test(dataset)
        selected_data, selected_y = create_dataset(data_list,y_list)
        
        print('saving in',path)
        with open(path+dataName+'_data', "wb") as s_data:   #Pickling
            pickle.dump(selected_data, s_data)

        with open(path+dataName+'_y', "wb") as s_y:   #Pickling
            pickle.dump(selected_y, s_y)
        
        
    return 
    
    
if __name__ == "__main__":
    path='/afs/le.infn.it/user/c/centonze/WQeMS/data/'

    main(path)