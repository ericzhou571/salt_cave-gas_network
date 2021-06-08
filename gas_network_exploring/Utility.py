#Utility
import Visualisierungen
import prettytable
import pandas as pd
import json
import re 
from bokeh.io import output_notebook, show
from bokeh.models import GMapOptions,HoverTool,ColumnDataSource
from bokeh.plotting import gmap
import bokeh.palettes as bp

def string2list(string,with_None=True):
    p = re.compile('(?<!\\\\)\'')
    string=p.sub('\"', string)
    
    if with_None:
        p2=re.compile('None')
        string=p2.sub('\"None\"', string)

    return json.loads(string)



#create a report table about components in seven dataset
def check(dataset,how='single'):
    if how=='single':
        tables=prettytable.PrettyTable(['Component','Count'])
        
        for key in dataset.keys():
            tables.add_row([key,len(dataset[key])])
        print(tables)
    
    else:
        check_report={'component':['BorderPoints',
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
    'Storages' ]}
        for key in dataset.keys():
            check_report[key]=[]
            for i in range(len(check_report['component'])):
                try:
                    check_report[key].append(len(dataset[key].frame_dict[check_report['component'][i]]))
                except:
                    check_report[key].append('')

        return pd.DataFrame(check_report)



def limit2EU(table):
    EU=['BE','BG','CZ','DK','DE','EE','IE','EL','ES','FR','HR','IT'
    ,'CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','IS','LI','NO','CH','XX']
    
    #create mask in two different Situation 
    if re.search('\[*\]',table['country_code'][0]):
    #list like country code    
        mask=table.country_code.apply(lambda x:string2list(x))
        start=mask.apply(lambda x: x[0])
        end=mask.apply(lambda x: x[1])
        mask=start.isin(EU)|end.isin(EU)

    else:
    #single country code
        mask=table.country_code.isin(EU)

    new_table=table.loc[mask,:].copy()
    new_table.reset_index(inplace=True)

    return new_table


def report_capacity(dataset,patterns=['M_m3'],report_model=False,check_model=False):
    '''
    return a Dictionary of Dictionary
    like final_report[key]={'EU_report':EU_report,
                            'country_report':country_report,
                            'crossboard_report':crossboard_report}
    final_report also a dict
    '''
    #will print components info about current dataset
    if check_model:
        check(dataset)
    
    final_report={}

    for key in dataset.keys():

        #++++++++++++++Part 1: Set Variable++++++++++++++++++++++++++++

        #for each table  
        current=dataset[key].copy()
        current['amount']=1

        #only EU countries
        current=limit2EU(current)

        if report_model:
            print(key)
            print('------------------------------')
        
        #which attribute to choose
        target=['amount']

        list_country_code=False

        #only use when list ike country code 
        inside=None
        crossboard=None

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++++Part 2: Prepare Data++++++++++++++++++++++++++++


        #--------------check if list [] like country code--------------
        if re.search(r'\[*\]',current['country_code'][0]):
            list_country_code=True
            p = re.compile('(?<!\\\\)\'')

            current.country_code=current.country_code.apply(lambda x:p.sub('\"', x) )

            current.country_code=current.country_code.apply(lambda x : json.loads(x))

            current['start']=current.country_code.apply(lambda x: x[0])
            current['end']=current.country_code.apply(lambda x: x[1])


        #------------search matched column names and save in target----
        for x in current.columns:

            #search key words about capacity
            #eu level
            for pattern in patterns:
                if re.search(pattern,x):
                    target.append(x)

        #print(target)

        #--------------check if there is match column name---------------
        if not target:
        #nothing match
            if report_model:
                print('-------------------------------\n\n')
            continue
        

       #---------------replace missing value(None) with 0----------------
        for column in target:
            current[column]=current[column].replace('None',0)

       #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
       #++++++++++++++Part 3: create report++++++++++++++++++++++++++++++
       
       #also need to return Components name (key)
       #report have three type:
       #country report
       #EU report
       #crossborad report(if only point,then this is str NULL)
       

       #--------------List like country code-----------------------------
        if list_country_code:
            print(list(current.columns))
            #EU

            #capacity per length_km
            try:
                current['length_km']=current[column].replace('None',1)
                current['max_cap_M_m3_per_d']=current['max_cap_M_m3_per_d']/current['length_km']
                #current.drop('length_km',axis=1,inplace=True)
            except KeyError:
                print(key,current.columns)
                

            EU_report=current[target].sum()
            if report_model:        
                print('eu\n',EU_report)
                
                print('+++++++++++++++++++++++++++++++++')
            
            #country
            inside=current[current.start==current.end]
            inside=inside.drop(['end'],axis=1)
            crossboard=current[current.start!=current.end]
            target.append('start')

            country_report=inside[target].groupby(by='start').sum()

            if report_model:
                print('inside country\n',country_report)
                print('+++++++++++++++++++++++++++++++++')

            #crossboard
            target.append('end')
            crossboard=crossboard[target]
            crossboard['pipenumber']=1

            crossboard_report=crossboard.groupby(by=['start','end']).sum()
            if report_model:
                print('crossboard\n',crossboard_report)
                print('-------------------------------\n\n')

        #--------------normal country code-----------------------------    
        else:
            #eu  
            EU_report=current[target].sum()
            if report_model:      
                print('eu\n',)
                
                print('+++++++++++++++++++++++++++++++++')

            #country
            target.append('country_code')
            country_report=current[target].groupby(by='country_code').sum()
            if report_model:
                print('country',country_report)
                print('-------------------------------\n\n')
            
            crossboard_report='NULL'

        #print(key)
        final_report[key]={'EU_report':EU_report,'country_report':country_report,'crossboard_report':crossboard_report}
    
    return final_report


def draw_several_dataset(df1,df2):

    map_options = GMapOptions(lat=51.10, lng=12, map_type="roadmap", zoom=6)

    #google api key
    api_key = "AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY"

    p = gmap(api_key, map_options, title="EMAP")

    TOOLTIPS=[('capacity','@max_cap_M_m3_per_d'),
             ('pipe_name','@name_without_number'),
             ('from','@from'),
             ('to','@to'),
             ('capacity_entsog','@Capacity_hourly')]

    p.add_tools( HoverTool(tooltips=TOOLTIPS))


    source=ColumnDataSource(df1[['long','lat']])
    source2=ColumnDataSource(df2[['long','lat']])


    p.multi_line('long',
                 'lat',
                 color='white',
                 line_width=2,source=source)
    p.multi_line('long',
                 'lat',
                 color='yellow',
                 line_width=1,source=source2)
    print('df1 white, df2 yellow')
    show(p)

    