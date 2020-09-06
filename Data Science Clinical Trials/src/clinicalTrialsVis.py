# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 22:18:28 2020

@author: Savitri
"""

'''
    The below code will generate dynamic graph with their corresponding basic information. Please check the graphs here
    http://127.0.0.1:54545/
'''
import matplotlib.pyplot as plt
import pandas as pd


# pre-processing
def preprocessing():
    # read csv data file
    df1 = pd.read_csv('..\\data\\ctVis.csv', encoding='iso-8859-1')
    print(df1.info())
    df1['Primary Completion Date'].dropna(inplace=True)
    df1.sort_values(by=['Primary Completion Date'], inplace=True, ascending=False)
    df1['Intervention'] = sorted(df1['Intervention'], reverse=True)
    df1['Completion Date'] = pd.to_datetime(df1['Primary Completion Date'])
    indexnames = df1[df1['PCD Grade'] == 0].index
    df1.drop(indexnames, inplace=True)
    statusnames = df1[df1['Status'] == 'Suspended'].index
    df1.drop(statusnames, inplace=True)
    statusnames2 = df1[df1['Status'] == 'Trial Withdrawn'].index
    df1.drop(statusnames2, inplace=True)
    return df1


import plotly.express as px


# visualization
def display(df1):
    fig = px.scatter(df1, x='Completion Date', y='Intervention', range_x=['2019-12-01', '2020-12-31'],
                     color='PCD Grade', hover_data=['Registration Date', 'Trial ID'], width=1200, height=800)
    plt.xticks(rotation=90)
    fig.update_layout(title={
        'text': "Clinical trails based on COVID-19: Number of trials (" + str(len(df1)) + ')',
        'y': 1.0,
        'x': 0.6,
        'xanchor': 'center',
        'yanchor': 'top'
    })

    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    # display the graph
    fig.show()
    # save the graph as html
    fig.write_html("pcd.html")


# run the methods
if __name__ == '__main__':
    df1 = preprocessing()
    display(df1)
