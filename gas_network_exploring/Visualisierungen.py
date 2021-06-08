import pandas as pd
import json
import re
from bokeh.io import output_notebook, show
output_notebook()
from bokeh.models import GMapOptions
from bokeh.plotting import gmap
from pathlib import Path
#import bokeh.palettes as bp

from bokeh.io import output_notebook, show
output_notebook()
from bokeh.models import GMapOptions
from bokeh.plotting import gmap
import json
from prettytable import PrettyTable



class GridDataset:
    possible_data_name=['BorderPoints',
    'Compressors',
    'ConnectionPoints',
    'Consumers',
    'EntryPoints',
    'InterConnectionPoints',
    'LNGs',
    'Nodes',
    'PipeLines',
    'PipePoints',
    'PipeSegments',
    'Processes',
    'Productions',
    'Storages' ]
    
    seven_dataset=['EMAP_Raw','GIE_Raw','IGG','INET_Filled','INET_Raw','LKD_Raw','NO_Raw','IGGI','IGGINL','IGGIN']
    #set colors pattle
    #color_list_point=bp.Accent7
    color_list_point=['red','blue','yellow','green','orange','purple','brown','pink','black','gray','white',
                     'green','orange','purple']
    #color_list_line=bp.Spectral[7]
    color_list_line=['pink','black','white','orange','purple','brown','red','blue','yellow','green','gray',
                    'green','orange','purple']
    
    def __init__(self,dataset_name,quite=False):
        self.dataset_name=dataset_name
        self.frame_dict={}
        p=Path('.')
        #load all data
        for data in self.possible_data_name:
            data_path=list(p.glob('**/'+dataset_name+'/**/*'+data+'.csv'))
            if data_path:
                current_table=pd.read_csv(data_path[0],sep=';')
            else:
                continue
            
            
            self.frame_dict[data]=current_table
        if 'PipeSegments' in self.frame_dict.keys():
            current_table=self.frame_dict['PipeSegments']
            current_table=self.unpack_param(current_table)
            current_table['lat']=current_table.lat.apply(json.loads)
            current_table['long']=current_table.long.apply(json.loads)
            self.frame_dict['PipeSegments']=current_table
            #self.frame_dict['PipeSegments']=self.unpack_param(current_table)
        if not quite:
            print('successfully load dataset:',self.dataset_name,' it have components: ',list(self.frame_dict.keys()))
            
    def unpack_param(self,df):
        if 'param' in df.columns:
            new = df.param.apply(self.string2list)
            return df.merge(pd.DataFrame(list(new)), left_index=True, right_index=True)
        else:
            return df        
    
    def string2list(self,string,with_None=True):
        p = re.compile('(?<!\\\\)\'')
        string=p.sub('\"', string)

        if with_None:
            p2=re.compile('None')
            string=p2.sub('\"None\"', string)

        return json.loads(string)
    

    def ConvertString2Dict(self,pd_series):
        '''
            depend on library:
                re
                pandas as pd
                json
            input:
                is a series of dict like string : '{'a':1,'b':2}'
            return:
                return a dataframe the same index like input pandas series
        '''
        #replace \' with \" . this is a problem by using json.loads
       # p = re.compile('(?<!\\\\)\'')
       # pd_series=pd_series.apply(lambda x: p.sub('\"', x))

        #None will be a problem when convert to dict
        #Use '\"None\"' to replace
        # replace not work ,dont know why just use re
       # p2=re.compile('None')
       # pd_series=pd_series.apply(lambda x: p2.sub('\"None\"', x))
        #convert to a series of dictionary object
        pd_series=pd_series.apply(string2list)
        return pd.DataFrame(list(pd_series))


    ##part for plot
    def add_plot_components(self,bokeh_gmap_object,components=[],point_size=10):
        #components must be ilterable
        
        if not components:
            components=self.frame_dict.keys()
            
        #use to give different component with different shape
        count=0
        
        #output message
        final_message={'Dataset':self.dataset_name,'Line_data':'-','Point_color':'-','Line_color':'-','Capacity_data':'-'}
        
        for component in components :
            try:
                current_table=self.frame_dict[component]
                #add component name
                current_table['component_name']=component
            except:
                #without this component
                #message
                final_message[component]='-'
                count+=1
                continue
       
            if type(current_table['long'][0])==list:
                #plot line
                # line_width is realtiv cpapcity of pipeline
                
                #output message
                final_message['Line_data']='Yes'
                

                fill_color=self.color_list_line[self.seven_dataset.index(self.dataset_name)]
                final_message['Line_color']=fill_color
                try:
                    bokeh_gmap_object.multi_line(current_table['long'],current_table['lat'],color=fill_color,line_width=2+10*self.minMaxScaler(current_table['max_cap_M_m3_per_d']))

                    #output message
                    final_message['Capacity_data']='Yes'
                except:
                    
                    #output message
                    final_message['Capacity_data']='No'
                    bokeh_gmap_object.multi_line(current_table['long'],current_table['lat'],color=fill_color,line_width=1)
                
            else:
                #diamond
                #cross
                #hex
                #inverted_triangle
                #plus
                #square_cross
                #circle
                #asterisk
                
                #not have so many color if component too much may repeat
                fill_color=self.color_list_point[self.seven_dataset.index(self.dataset_name)%len(self.color_list_point)]
                final_message['Point_color']=fill_color
                
                #change to other shape
                if count==0:
                    bokeh_gmap_object.circle(x="long", y="lat", size=point_size,color=fill_color, fill_color=fill_color, fill_alpha=1, source=current_table)
                    #output message
                    final_message[component]='circle'
                elif count==1:
                    bokeh_gmap_object.asterisk(x="long", y="lat", size=point_size,color=fill_color, fill_color=fill_color, fill_alpha=1, source=current_table)
                    #output message
                    final_message[component]='asterisk'
                elif count==2:
                    bokeh_gmap_object.diamond(x="long", y="lat", size=point_size,color=fill_color, fill_color=fill_color, fill_alpha=1, source=current_table)
                    #output message
                    final_message[component]='diamond'
                elif count==3:
                    bokeh_gmap_object.hex(x="long", y="lat", size=point_size,color=fill_color, fill_color=fill_color, fill_alpha=1, source=current_table)
                    #output message
                    final_message[component]='hex'
                elif count==4:
                    bokeh_gmap_object.square_cross(x="long", y="lat", size=point_size,color=fill_color, fill_color=fill_color, fill_alpha=1, source=current_table)
                    #output message
                    final_message[component]='square_cross'
                elif count==5:
                    bokeh_gmap_object.inverted_triangle(x="long", y="lat", size=point_size,color=fill_color, fill_color=fill_color, fill_alpha=1, source=current_table)
                    #output message
                    final_message[component]='inverted_triangle'
                elif count>=6:
                    bokeh_gmap_object.cross(x="long", y="lat", size=point_size,color=fill_color, fill_color=fill_color, fill_alpha=1, source=current_table)
                    #output message
                    final_message[component]='cross'
                    
                count+=1
        return final_message
    def minMaxScaler(self,series):
        return (series-min(series))/max(series)



class Visual:
    
    #check if choosed dataset
    dataset_choosed=0
    #check if choosed components
    components_choosed=0
    #possible value
    possible_datasets=['EMAP_Raw','GIE_Raw','IGG','INET_Filled','INET_Raw','LKD_Raw','NO_Raw','IGGI','IGGINL','IGGIN']
    
    possible_components=['BorderPoints',
    'Compressors',
    'ConnectionPoints',
    'Consumers',
    'EntryPoints',
    'InterConnectionPoints',
    'LNGs',
    'Nodes',
    'PipeLines',
    'PipePoints',
    'PipeSegments',
    'Processes',
    'Productions',
    'Storages' ]
    
    def __init__(self,quite=False):
        self.dataset={}
        for dataset in self.possible_datasets:
            #print(dataset)
            #load data and save in object
            self.dataset[dataset]=GridDataset(dataset,quite=True)
        
        #-------------------------prepare general plot parameters---------------
        map_options = GMapOptions(lat=51.10, lng=12, map_type="roadmap", zoom=6)


        #google api key
        api_key = "AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY"

        self.p = gmap(api_key, map_options, title="EMAP")
        #------------------------------------------------------------------------
        
        if not quite:
            self.check()
            self.helps()
        
    def plot(self):
        
        if self.components_choosed==0:
            print('please choose components')
            return
        
        line_table_columns=['Dataset','Line_data','Capacity_data','Line_color']
        line_table=PrettyTable(line_table_columns)
        
        point_table_columns= self.components_choosed.copy()
        point_table_columns.insert(0,'Dataset')
        point_table_columns.append('Point_color')
        if 'PipeSegments'in point_table_columns: 
            point_table_columns.remove('PipeSegments')
        
        point_table=PrettyTable(point_table_columns)
        
        if self.dataset_choosed==0:
            print('dataset not choosed, please choose dataset!')
            return
        
        else:
            for table in self.dataset_choosed:
                #incept of messages 
                message=self.dataset[table].add_plot_components(self.p,self.components_choosed)
                line_table.add_row([message[x] for x in line_table_columns])
                point_table.add_row([message[x] for x in point_table_columns])
        
        print('pipeline information')
        print(line_table)
        print('point information')
        print(point_table)
        show(self.p)
    
    def choose_dataset(self,list_dataset):
        if type(list_dataset)==list:
            self.dataset_choosed=list_dataset
            print('dataset successful!')
        else:
            print('input is not a list, please give a list')
    
    def choose_component(self,list_component):
        if type(list_component)==list:
            self.components_choosed=list_component
            print('component successful!')
        else:
            print('input is not a list, please give a list')
        
    def reset(self):
        #-------------------------prepare general plot parameters---------------
        map_options = GMapOptions(lat=51.10, lng=12, map_type="roadmap", zoom=6)


        #google api key
        api_key = "AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY"

        self.p= gmap(api_key, map_options, title="EMAP")
        
        self.components_choosed=0
        self.dataset_choosed=0
        #------------------------------------------------------------------------
        self.check()

    
    def helps(self,simple=False):
        if not simple:
            print('\n\nInstruction\n---------------')
            print('choose dataset')
            print('please choose dataset from: \n\n*********************\n\n\n'
                  ,self.possible_datasets,'\n\n\n*********************\n')
            print('Example: use your object.choose_dataset([\'EMAP_Raw\',\'GIE_Raw\'])','\n')
            print('--------------')
            print('choose component')
            print('please choose component from:\n\n\n*********************\n\n',
                  self.possible_components,'\n\n\n*********************\n\n\n')
            print('Example: use your object.component([\'Nodes\',\'Nodes\'])','\n')
        else:
            print(self.possible_datasets)
            print('#####################')
            print(self.possible_components)
        
        
    def check(self):
        if self.dataset_choosed==0:
            print('Dataset\n ----------------')
            print('Status: Waiting input!\n')
        else:
            print('Status: Successful!\n')
            print(self.dataset_choosed)
        if self.components_choosed==0:
            print('\nComponent\n ----------------')
            print('Status: Waiting input!\n')
        else:
            print('Status: Successful!\n')
            print(self.components_choosed)