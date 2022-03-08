#modules!
import os
import tarfile
from six.moves import urllib
from sklearn.datasets import fetch_openml
import pandas as pd

# another trick to get total uri.
# scriptpath = os.path.realpath(__file__)

download_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
path = "datasets/housing"
housing_url_base = download_url + path + "/housing.tgz";


def saveFile(dataframe, uri , idx=False):
    dataframe.to_csv(path_or_buf=uri, index=idx)


def getFromOpenML(database,version=1,ospath="online_download/", download = False, save=True):

    instances,instances_labels = 0,0

    if download:
        if not os.path.isdir(ospath):
            os.makedirs(ospath)

        X,Y = fetch_openml(database, version=version, return_X_y=True)
        instances = pd.DataFrame.from_records(X)
        instances_labels = pd.DataFrame.from_records(Y)
        inst_cols = instances.columns

        newNames = []
        for i in inst_cols:
            name= inst_cols[i]

            newNames.append( "".join([ "col_",str(name) ]) )


        instances.columns = newNames

        labels_cols = instances_labels.columns
        newNames = []
        for i in labels_cols:
            name= labels_cols[i]

            newNames.append( "".join(["label_", str(name) ]))



        instances_labels.columns = newNames
    else:
        to_load = "".join([ospath,database,".csv"])
        return pd.read_csv( to_load)
    #print(instances)
    #print(instances_labels)
    full_database = pd.concat([instances, instances_labels], axis=1)
    full_uri_to_save = ""
    full_uri_to_save = full_uri_to_save.join([ospath,database,".csv"])
    if save:
        full_database.to_csv(path_or_buf=full_uri_to_save, index=False)

    return full_database


def getFullURI(fl):
    return  os.path.join(os.getcwd() , fl )

def getData( housing_url = housing_url_base, housing_path=path):

    print("Downloading from " + housing_url_base)

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    tgz_path= os.path.join( os.getcwd() , tgz_path)

    returned = urllib.request.urlretrieve(housing_url,tgz_path)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
