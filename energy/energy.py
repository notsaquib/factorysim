import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from numpy.distutils.system_info import xft_info


class EnergyHandler:
    def __init__(self):
        self.energy = 0
        self.market_db: pd.DataFrame = pd.DataFrame()

    def load_market(self, market_csv):
        '''
        This method loads the energy market data into pandas dataframe
        :param market_csv: csv file with energy market data
        :return:
        '''
        market_db = pd.read_csv(market_csv, sep=';')
        start_time = []
        end_time = []
        for i in range(len(market_db)):
            start_time.append(
                datetime.strptime(market_db.iloc[i].Date + ',' + market_db.iloc[i].Start, '%b %d, %Y,%I:%M %p'))
            end_time.append(
                datetime.strptime(market_db.iloc[i].Date + ',' + market_db.iloc[i].End, '%b %d, %Y,%I:%M %p'))
        market_db['Start'] = start_time
        market_db['End'] = end_time
        self.market_db = market_db

    def get_price_by_time(self, current_time):
        '''
        This method gets the energy price at a given time
        :param current_time: the given time
        :return: energy price
        '''
        # for i in range(len(self.market_db)):
        #     start_time = self.market_db.iloc[i].Start
        #     End_time = self.market_db.iloc[i].End
        #     if current_time > start_time and current_time < End_time:
        #         return self.market_db['Germany/Luxembourg [€/MWh] Original resolutions'].iloc[i]
        try:
            index=self.market_db['End'].searchsorted(current_time, side='left')
            return self.market_db['Germany/Luxembourg [€/MWh] Original resolutions'].iloc[index]
        except:
            return 999999999999999999

    def plot_prices(self):
        fig=px.line(self.market_db,x='Start',y='Germany/Luxembourg [€/MWh] Original resolutions')
        fig=go.Scatter(x=self.market_db['Start'],y=self.market_db['Germany/Luxembourg [€/MWh] Original resolutions'])
        return fig