import streamlit as st

# import altair as alt
import plotly.express as px

import numpy as np
import pandas as pd
import datetime
import xlrd

# from datetime import datetime
# from dateutil.relativedelta import relativedelta

# import html2text
# import os
# import json
# import lxml
import itertools
from itertools import combinations

from pyvis.network import Network
from collections import Counter

import streamlit.components.v1 as components

# from st_aggrid import AgGrid


st.set_page_config('PF EMEA deals with insto',layout='wide')

st.title('EMEA project finance deals with institutional investors')

### Infralogic data

# path = 'C:/Users/matth/Documents/pythonprograms/Project_Finance_EMEA'
#
# folderlist = os.walk('C:/Users/matth/Documents/pythonprograms/Project_Finance_EMEA')
# folderlist = [folder[0] for folder in folderlist]
# folderlist = folderlist[1:]

# st.write(folderlist)

# maxidf = pd.DataFrame()
#
#
# for folder in folderlist:
#     filelist = os.listdir(folder)
#     filelist = [file for file in filelist]
#
#
#     for file in filelist:
#         with open(folder+'/'+file,encoding='utf-8') as data_file:
#             data = json.load(data_file)
#         nbdeals = len(data['transactions'])
#
#         id = [data['transactions'][i]['details']['id'] if not None else 'N/A' for i in range(nbdeals)]
#         name = [data['transactions'][i]['details']['name'] if not None else 'N/A' for i in range(nbdeals)]
#         region = [data['transactions'][i]['details']['regions'][0] if not None else 'N/A' for i in range(nbdeals)]
#         sector = [data['transactions'][i]['details']['sectors'][0] if not None else 'N/A' for i in range(nbdeals)]
#         currency = [data['transactions'][i]['details']['currency'] if not None else 'N/A' for i in range(nbdeals)]
#         fundings = [data['transactions'][i]['details']['fundings'] if not None else 'N/A' for i in range(nbdeals)]
#         countries = [data['transactions'][i]['details']['countries'][0]['name'] if not None else 'N/A' for i in range(nbdeals)]
#         states = [data['transactions'][i]['details']['countries'][0]['states'] if not None else 'N/A' for i in range(nbdeals)]
#         subsectors = [data['transactions'][i]['details']['subSectors'][0] if not None else 'N/A' for i in range(nbdeals)]
#         description = [data['transactions'][i]['details']['description'] if not None else 'N/A' for i in range(nbdeals)]
#         ratios = [data['transactions'][i]['details']['financeRatios'] if not None else 'N/A' for i in range(nbdeals)]
#         dominantregion = [data['transactions'][i]['details']['dominantRegion'] if not None else 'N/A' for i in range(nbdeals)]
#         dominantsector = [data['transactions'][i]['details']['dominantSector'] if not None else 'N/A' for i in range(nbdeals)]
#         dominantcountry = [data['transactions'][i]['details']['dominantCountry'] if not None else 'N/A' for i in range(nbdeals)]
#         transactiontype = [data['transactions'][i]['details']['transactionType'] if not None else 'N/A' for i in range(nbdeals)]
#         dominantsubsector = [data['transactions'][i]['details']['dominantSubSector'] if not None else 'N/A' for i in range(nbdeals)]
#         transactionstatus = [data['transactions'][i]['details']['transactionLifecycle']['transactionStatus'] if not None else 'N/A' for i in range(nbdeals)]
#         transactioncharacteristicsPPP = [data['transactions'][i]['details']['transactionCharacteristics']['PPP'] if not None else 'N/A' for i in range(nbdeals)]
#         transactioncharacteristicsutility = [data['transactions'][i]['details']['transactionCharacteristics']['utility'] if not None else 'N/A' for i in range(nbdeals)]
#
#
#         dealdict = {'id':id,'Name':name,'region':region,'sector':sector,'currency':currency,'fundings':fundings,'countries':countries,'states':states,'subsectors':subsectors,
#             'description':description,'ratios':ratios,'dominantRegion':dominantregion,'dominantsector':dominantsector,'dominantcountry':dominantcountry,
#             'transactiontype':transactiontype,'dominantsubsector':dominantsubsector,'transactionstatus':transactionstatus,
#             'PPP':transactioncharacteristicsPPP,'Utility':transactioncharacteristicsutility}
#
#         dealdf = pd.DataFrame(dealdict)
#
#         # df.to_excel('natixis_deals.xlsx')
#
#         name = []
#         type = []
#         tenor = []
#         amount = []
#         role = []
#         lenders = []
#         dateadded = []
#         allocation = []
#         estimatedAlloc = []
#         estimatedAllocUSD = []
#
#         ratings = []
#         retired = []
#         borrower = []
#         comments = []
#         monoline = []
#         debtclass = []
#         margininfo = []
#         marginvalue = []
#         facilitytype = []
#         maturity = []
#         lifecyclepoint = []
#         amountusd = []
#
#
#         for transaction in data['transactions']:
#             for tranche in transaction['details']['fundings']:
#                 try:
#                     for lender in tranche['lenders']:
#                         try:
#                             name.append(transaction['details']['name'])
#                         except:
#                             name.append(None)
#                         try:
#                             type.append(tranche['type'])
#                         except:
#                             type.append(None)
#                         try:
#                             tenor.append(tranche['tenor'])
#                         except:
#                             tenor.append(None)
#                         try:
#                             amount.append(tranche['amount'])
#                         except:
#                             amount.append(None)
#                         try:
#                             role.append(lender['role'])
#                         except:
#                             role.append(None)
#                         try:
#                             lenders.append(lender['lender']['name'])
#                         except:
#                             lenders.append(None)
#                         try:
#                             dateadded.append(lender['dateAdded'])
#                         except:
#                             dateadded.append(None)
#                         try:
#                             allocation.append(lender['allocation'])
#                         except:
#                             allocation.append(None)
#                         try:
#                             estimatedAlloc.append(lender['estimatedAllocation'])
#                         except:
#                             estimatedAlloc.append(None)
#                         try:
#                             estimatedAllocUSD.append(lender['estimatedAllocationUSD'])
#                         except:
#                             estimatedAllocUSD.append(None)
#                         try:
#                             ratings.append(tranche['ratings'])
#                         except:
#                             ratings.append(None)
#                         try:
#                             borrower.append(tranche['borrower'])
#                         except:
#                             borrower.append(None)
#                         try:
#                             comments.append(tranche['comments'])
#                         except:
#                             comments.append(None)
#                         try:
#                             monoline.append(tranche['monoline'])
#                         except:
#                             monoline.append(None)
#                         try:
#                             debtclass.append(tranche['debtClass'])
#                         except:
#                             debtclass.append(None)
#                         try:
#                             margininfo.append(tranche['marginInfo'])
#                         except:
#                             margininfo.append(None)
#                         try:
#                             marginvalue.append(tranche['marginValue'])
#                         except:
#                             marginvalue.append(None)
#                         try:
#                             facilitytype.append(tranche['facilityType'])
#                         except:
#                             facilitytype.append(None)
#                         try:
#                             maturity.append(tranche['maturityDate'])
#                         except:
#                             maturity.append(None)
#                         try:
#                             lifecyclepoint.append(tranche['lifeCyclePoint'])
#                         except:
#                             lifecyclepoint.append(None)
#                         try:
#                             amountusd.append(tranche['amountUSD'])
#                         except:
#                             amountusd.append(None)
#                 except:
#                     pass
#
#         ticketdf = pd.DataFrame([name,type,tenor,amount,role,lenders,dateadded,allocation,estimatedAlloc, estimatedAllocUSD,ratings,retired,borrower,comments,monoline,debtclass,margininfo,marginvalue,facilitytype,maturity,lifecyclepoint,amountusd]).transpose()
#         ticketdf.columns = ['Name','Type','Tenor','Amount','Role','Lender','Date added','Allocation','Estimated Allocation','Estimated Allocation USD','Ratings','Retired','Borrower','Comments','Monoline','Debt Class','Margin info','Margin value','Facility type','Maturity','Life cycle point','Amount USD']
#
#
#         bigdf = pd.merge(dealdf,ticketdf,on='Name')
#         bigdf.drop('fundings',axis=1,inplace=True)
#
#         maxidf = pd.concat([maxidf,bigdf])
#
# bigdf = maxidf

def formatcol(string):
    try:
        float(string)
        return float(string)
    except:
        return None


def rightname(string):
    if string == 'Natixis (Groupe BPCE)':
        return 'Natixis'
    elif 'SCOR' in string:
        return 'SCOR'
    else:
        return string

def isinsto(lender):
    if lender in instolist:
        return True
    else:
        return False

def iscountry(country):
    if country in countrylist:
        return True
    else:
        return False


@st.cache(allow_output_mutation=True)
def getdata():

    def isinsto(lender):
        if lender in instolist:
            return True
        else:
            return False

    def iscountry(country):
        if country in countrylist:
            return True
        else:
            return False


    bigdf = pd.read_csv('EMEA_deals.csv').iloc[:,1:]
    bigdf['Amount'] = bigdf['Amount'].apply(formatcol)

    bigdf.index.name=None


    bigdf['Lender']=bigdf['Lender'].apply(rightname)

# bigdf

# lenderlist = pd.DataFrame(bigdf['Lender'].unique())
# lenderlist.to_excel('lender_list.xlsx')

    lenderlist = pd.read_excel('lender_list.xlsx').iloc[:,1:]

    lenderlist.columns=['Lender','Is insto','Development agency']
    instolist = lenderlist[lenderlist['Is insto']=='x']

    dfwithinsto = pd.merge(bigdf,instolist,left_on='Lender',right_on='Lender')
    # dfwithinsto

    instolist = instolist['Lender'].tolist()

    idlist = pd.DataFrame(dfwithinsto['id'].unique())
    idlist.columns=['id']

    dealswithinsto = pd.merge(bigdf,idlist,left_on='id',right_on='id')

    countrylist = pd.read_excel('countries.xlsx').iloc[:,1:]
    countrylist = countrylist[countrylist['In']=='x']
    countrylist.columns=['Country','In']
    countrylist = countrylist['Country'].unique().tolist()


    dealswithinsto['Is insto'] = dealswithinsto['Lender'].apply(isinsto)
    dealswithinsto['Is Europe'] = dealswithinsto['countries'].apply(iscountry)

    dealswithinsto = dealswithinsto[dealswithinsto['Is Europe']==True]

    instolist = dealswithinsto[dealswithinsto['Is insto']==True]['Lender'].unique().tolist()
    banklist = dealswithinsto[dealswithinsto['Is insto']==False]['Lender'].unique().tolist()

    hasinsto = dealswithinsto[(dealswithinsto['Is Europe']==True)&(dealswithinsto['Is insto']==True)]

    return dealswithinsto,instolist,banklist,hasinsto


dealswithinsto,instolist,banklist,hasinsto = getdata()


st.subheader('Deals with at least one insto')


chartlist = ['sector > subsector > country > deal name','country > sector > subsector > deal name','sector > lender > country > subsector > deal name']
selectchart = st.selectbox('Select chart',chartlist,0)

if selectchart == 'sector > subsector > country > deal name':
    fig = px.treemap(hasinsto,path=['sector','subsectors','countries','Name'],color='sector')
elif selectchart == 'country > sector > subsector > deal name':
    fig = px.treemap(hasinsto,path=['countries','sector','subsectors','Name'],color='countries')
else:
    fig = px.treemap(hasinsto,path=['sector','Lender','countries','subsectors','Name'],color='Lender')
st.plotly_chart(fig)

st.subheader('Deals by insto')



instolist = dealswithinsto[(dealswithinsto['Is Europe']==True)&(dealswithinsto['Is insto']==True)]['Lender'].unique().tolist()
instolist.sort()

instotickets = dealswithinsto[(dealswithinsto['Is Europe']==True)&(dealswithinsto['Is insto']==True)].drop_duplicates(subset=['Name','Lender'])

chartlist = ['lender > sector > deal name','lender > country > sector > subsector > deal name']
selectchart = st.selectbox('Select chart',chartlist,0,key=1)

if selectchart == 'lender > sector > deal name':
    fig = px.treemap(instotickets,path=['Lender','sector','Name'],color='countries')
else:
    fig = px.treemap(instotickets,path=['Lender','countries','sector','subsectors','Name'],color='countries')
st.plotly_chart(fig)

selectinsto = st.selectbox('Select insto',instolist)
dealsbyinsto = instotickets[instotickets['Lender']==selectinsto]
st.write(dealsbyinsto)


# st.write(dealswithinsto[(dealswithinsto['Is Europe']==True)&(dealswithinsto['Is insto']==True)]['Lender'].value_counts())

# dealcount = instotickets.groupby(by=['Lender'])['Name'].count().sort_values(ascending=False)
# st.write(dealcount)

st.header('Deals with Nord/LB')

nbdeals = dealswithinsto[dealswithinsto['Lender'].str.contains('Norddeutsche')]
idlist = nbdeals['id'].unique().tolist()
nbdeals = nbdeals[nbdeals['id'].isin(idlist)]
nbinsto = dealswithinsto[(dealswithinsto['id'].isin(idlist))&(dealswithinsto['Is insto']==True)]['Lender'].unique().tolist()

st.subheader('Instos that took part in Nord/LB deals')
# st.write(dealswithinsto[(dealswithinsto['id'].isin(idlist))&(dealswithinsto['Is insto']==True)].drop_duplicates(subset=['Name','Lender']))
st.write(dealswithinsto[(dealswithinsto['id'].isin(idlist))&(dealswithinsto['Is insto']==True)].drop_duplicates(subset=['Name','Lender'])[['Lender','Name','sector','subsectors','countries']])

st.subheader('Nord/LB deals where instos took part')
st.write(nbdeals.drop_duplicates(subset=['Name']))

st.metric('Number of deals',len(idlist))
st.metric('Number of instos',len(nbinsto))


##### Network analysis #####

st.header('Network analysis')

st.subheader('Network Influencers')

idlist = dealswithinsto['id'].unique().tolist()

biglenderpairs = []
for id in idlist:
    df = dealswithinsto[dealswithinsto['id']==id]
    lenderlist = df['Lender'].unique().tolist()
    lenderlist.sort()
    lenderpairs = list(combinations(lenderlist,2))
    biglenderpairs.append(lenderpairs)

maxilist=[]
for sublist in biglenderpairs:
    maxilist+=sublist

countpair = []
for item in maxilist:
    countpair.append(maxilist.count(item))

def instoinpair(item):
    if item[0] in instolist:
        return True
    elif item[1] in instolist:
        return True
    else:
        return False

df = pd.DataFrame()
df['Pair']=maxilist
df['Count']=countpair
df['Insto in Pair']=df['Pair'].apply(instoinpair)
df['Member1']=df['Pair'].apply(lambda x:x[0])
df['Member2']=df['Pair'].apply(lambda x:x[1])


maxilistwithedges = []
countlist = df['Count'].tolist()
j=0
for i in maxilist:
    triplet = (i[0],i[1],countlist[j])
    if (i[0] in instolist or i[1] in instolist):
        maxilistwithedges.append(triplet)
    j=j+1
network = maxilistwithedges

# friends = pd.DataFrame(network,columns=['person1','person2'])

network = pd.DataFrame(network)
network.columns=['Institution 1','Institution 2','Number interactions']

def get_friends(data:pd.DataFrame,person_id):
    list1=data[data['Institution 1']==person_id]['Institution 2'].values.tolist()
    list2=data[data['Institution 2']==person_id]['Institution 1'].values.tolist()
    mylist = list(set(list1+list2))
    mylist.sort()
    return mylist

def get_num_friends(data:pd.DataFrame,person_id:int):
    return len(get_friends(data,person_id))


def get_num_friends_map(data:pd.DataFrame):
    all_people = list(set(data['Institution 1'].unique().tolist()).union(set(data['Institution 2'].unique().tolist())))
    all_people.sort()
    return {name:get_num_friends(network,name) for name in all_people}

def get_num_friends_of_a_person_friends(data:pd.DataFrame,person_id,num_friends_map:dict):
    friends = get_friends(data,person_id)
    return [num_friends_map[friend_id] for friend_id in friends]

def get_average_friends_of_a_person_friends(data:pd.DataFrame,person_id):
    num_friends_map = get_num_friends_map(network)
    num_friends_of_friends = get_num_friends_of_a_person_friends(data, person_id, num_friends_map)
    return np.mean(num_friends_of_friends)

# num_friends_map = get_num_friends_map(network)
# st.write(get_num_friends_of_a_person_friends(network,'Natixis',num_friends_map))
# st.write(get_average_friends_of_a_person_friends(network,'Natixis'))

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def get_friends_df(data:pd.DataFrame):
    placeholder = st.empty()
    progholder = st.empty()
    mybar = st.progress(0)

    i=1
    all_people = list(set(data['Institution 1'].unique().tolist()).union(set(data['Institution 2'].unique().tolist())))
    all_people.sort()
    num_friends=[]
    totalfiles=len(all_people)
    for person_id in all_people:
        num_friends.append({
            'Institution':person_id,
            'num_friends':get_num_friends(data,person_id),
            'avg_friends_of_friends':round(get_average_friends_of_a_person_friends(data,person_id),2)
        })
        with placeholder:
            st.write('Institution #{0} complete '.format(i)+'/ '+str(totalfiles)+'.')
        with progholder:
            pct_complete = '{:,.2%}'.format(i/totalfiles)
            st.write(pct_complete,' complete.' )
            try:
                mybar.progress(i/totalfiles)
            except:
                mybar.progress(1)
        i=i+1
    return pd.DataFrame(num_friends)

infludf = get_friends_df(network)
influencers = infludf[infludf['num_friends']>infludf['avg_friends_of_friends']]

def isinsto(string):
    if string in instolist:
        return True
    else:
        return False

influencers['Is insto']=influencers['Institution'].apply(isinsto)

st.subheader('Insto influencers')
st.write(influencers[influencers['Is insto']==True])

st.subheader('Bank influencers')
st.write(influencers[influencers['Is insto']==False])

influlist = influencers['Institution'].unique().tolist()

#NETWORK GRAPH

st.header('Generate network graph')


st.write('Red: Influencer bank, Green: Influencer insto, Grey: Follower bank, Blue: Follower insto')
# st.write(df.head(20))

net = Network(notebook=True, height='1500px',width='2500px')
people = list(set(df['Member1']).union(set(df['Member2'])))
people.sort()

colours = []
bankinsto = []
for i in people:
    if i in instolist and i in influlist:
        colours.append('#00FF00') #green
        bankinsto.append('Insto - Influencer - Green')
    elif i in instolist:
        colours.append('#0000FF') #blue
        bankinsto.append('Insto - Follower - Blue')
    elif i not in instolist and i in influlist:
        colours.append('#FF0000') #red
        bankinsto.append('Bank - Influencer - Red')
    else:
        colours.append('#BFC9CA') #grey
        bankinsto.append('Bank - Follower - Grey')

# st.write([(i,colour) for (i,colour) in zip(people,bankinsto)])

net.add_nodes(people,label=people,color=colours)
net.add_edges(maxilistwithedges)

net.repulsion(node_distance=600,spring_strength=0.025,damping=0.05)
# net.show_buttons(filter_=True)

folder = 'C:/Users/matth/Documents/pythonprograms/Project_Finance_EMEA'

try:
    path = '/tmp'
    # net.show('insto networks.html')
    net.save_graph(f'{path}/insto networks.html')
    HtmlFile = open(f'{path}/insto networks.html','r',encoding='utf-8')

except:
    path = '/html_files'
    # net.show('insto networks.html')
    net.save_graph('insto networks.html')
    HtmlFile = open('insto networks.html','r',encoding='utf-8')

components.html(HtmlFile.read(),height=800)


st.subheader('Which instos are the banks interacting with?')

selectbank = st.selectbox('Select bank',banklist)


df = network[(network['Institution 1']==selectbank)|(network['Institution 2']==selectbank)]
def choose1(a,b):
    if a==selectbank:
        return b
    else:
        return a

df.drop_duplicates(inplace=True)
df['Insto'] = df.apply(lambda x: choose1(x['Institution 1'],x['Institution 2']),axis=1)
df = df[['Insto','Number interactions']].sort_values('Number interactions',ascending=False)
df
