import polars as pl
import pandas as pd
import chardet
from pathlib import Path
from typing import List,IO, Dict, Any, Union, Optional


class DataIngest():
    '''
    This class is meant for data ingestion

    '''
    def __init__(self):
        pass 
    
    def read_csv(
            self,
            file_path:str,
            separator:Optional[str]=",",
            **kwargs
            )-> pl.DataFrame:
        """
        Read CSV file 
        
        Args:
            file_path(str) : File path for the csv file 
            delimiter(str) : delimiter for the csv file. Default ","
            **kwargs: Additional keywords arguments to pass for pandas.read_csv()
        Return:
            Pandas DataFrame
            
        """
        try:
            with open(file_path,'rb') as f:
                encoding = chardet.detect(f.read())
            try:
                df = pl.read_csv(file_path,encoding=encoding['encoding'],separator=separator,**kwargs)
                return df
            except UnicodeDecodeError:
                print("UnicodeDecodeError")
                return None
            except Exception as e:
                print(e)
                return None

        except FileNotFoundError:
            print(f"File not Found:{file_path}")
            return None
        except Exception as e:
            print(e)
            return None
    
    def read_excel(self,
                   file_path:str,
                   **kwargs
                   )-> pl.DataFrame:
        """
        This function is used to read excel file format. 
        
        Args:
            file_path(str): File path for the excel file 
            **kwargs : Additional keywords arguments to pass for pandas.read_excel

        Return:
            Pandas DataFrame.
        """
        try:
            df = pl.read_excel(source=file_path,**kwargs)
            return df 
        except Exception as e:
            print(e)
            return None
        except FileNotFoundError as e:
            print(e)
            return None
    
    def read_parquet(self,
                     file_path:str,
                     **kwargs
                     )-> pl.DataFrame:
        """
    This function is used to read parquet.

    Args:
        file_path(str): File path for the excel file 
        **kwargs : Additional keywords arguments to pass for pandas.read_parquet
    Return:
        Pandas DataFrame

        """
        try:
            df = pl.read_parquet(source=file_path,**kwargs)
            return df
        except Exception as e:
            print(e)
            return None
        except FileNotFoundError as e:
            print(e)
            return None
    
    def read_clipboard(self,separator: str = '\t',
                       **kwargs
                       )-> pl.DataFrame:
        """
        This function is to read clipboard
        """
        try:
            df = pl.read_clipboard(source=file_path,**kwargs)
            return df
        except Exception as e:
            print(e)
            return None
        except FileNotFoundError as e:
            print(e)
            return None
    
    def read_fwf(self,
                 file_path:str,
                 )->pd.DataFrame:
        """
        This function is used for pandas.read_fwf
        """
        try:
            df = pd.read_fwf(file_path,**kwargs)
            return df
        except Exception as e:
            print(e)
            return None
        except FileNotFoundError as e:
            print(e)
            return None

    def read_html(self,
                 file_path:str,
                 )->pd.DataFrame:
        """
        This function is used for pandas.read_html
        """
        try:
            df = pd.read_html(file_path,**kwargs)
            return df
        except Exception as e:
            print(e)
            return None
        except FileNotFoundError as e:
            print(e)
            return None

    def read_json(self,
                 source:str,
                 )->pl.DataFrame:
        """
        This function is used for pandas.read_html
        """
        try:
            df = pl.read_json(source,**kwargs)
            return df
        except Exception as e:
            print(e)
            return None
        except FileNotFoundError as e:
            print(e)
            return None

    def read_pickle(self,
                 file_path:str,
                 )->pd.DataFrame:
        """
        This function is used for pandas.read_html
        """
        try:
            df = pd.read_pickle(file_path,**kwargs)
            return df
        except Exception as e:
            print(e)
            return None
        except FileNotFoundError as e:
            print(e)
            return None

    def read_database(self,query: str,
                       **kwargs
                       )-> Union[pl.DataFrame,Any]:
        """
        This function is to read sql/database query
        """
        try:
            df = pl.read_database(query,**kwargs)
            return df 
        except Exception as e:
            print(e)
            return None
        except FileNotFoundError as e:
            print(e)
            return None
    
    def write_csv(self,
                  df:Any,
                  file_path:Union[str, Path, IO[bytes], bytes],
                  separator: Optional[str] = ',',
                  **kwargs)->Union[str,None]:
        """
        This is to write the polars DataFrame to csv file formate
        """
        try:
            df.write_csv(file_path,separator=separator,**kwargs)
            return None
        except Exception as e:
            print(e)
            return None

    def write_excel(self,
                  df:Any,
                  file_path:Union[str, Path, IO[bytes], bytes],
                  **kwargs)->Union[str,None]:
        """
        This is to write the polars DataFrame to csv file formate
        """
        try:
            df.write_excel(file_path,**kwargs)
            return None
        except Exception as e:
            print(e)
            return None   

    def to_csv(self,
                  df:Any,
                  file_path:Union[str, Path, IO[bytes], bytes],
                  separator: Optional[str] = ',',
                  **kwargs)->Union[str,None]:
        """
        This is to write the polars DataFrame to csv file formate
        """
        try:
            df.to_csv(file_path,sep=separator,**kwargs)
            return None
        except Exception as e:
            print(e)
            return None

    def to_excel(self,
                  df:Any,
                  file_path:Union[str, Path, IO[bytes], bytes],
                  **kwargs)->Union[str,None]:
        """
        This is to write the polars DataFrame to csv file formate
        """
        try:
            df.to_excel(file_path,**kwargs)
            return None
        except Exception as e:
            print(e)
            return None

    def to_pickle(self,
                  df:Any,
                  file_path:Union[str, Path, IO[bytes], bytes],
                  **kwargs)->Union[str,None]:
        """
        This is to write the polars DataFrame to csv file formate
        """
        try:
            df.to_pickle(file_path,**kwargs)
            return None
        except Exception as e:
            print(e)
            return None


di = DataIngest()
# df = di.read_csv(file_path="/run/media/shuga/Shuga/kaggle-playground-series-/train.csv")
df = pd.read_csv("/run/media/shuga/Shuga/kaggle-playground-series-/train.csv")
# df = di.read_excel("/home/shuga/Shuga/New Folder/All_ETS_Material_TC_GregWords.xlsx")
# df = pd.read_excel("/home/shuga/Shuga/New Folder/All_ETS_Material_TC_GregWords.xlsx")
print(df)
di.to_excel(df=df,file_path="/home/shuga/Shuga/New Folder/tharavu-dappa/src/tese12.xlsx")

