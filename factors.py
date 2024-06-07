import numpy as np

def Alpha088_to_df(df):
    # (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    def calculate(group):
        group['alpha88'] = (group['close']/(group['close'].shift(20)+1e-10)-1)*100
        return group
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha084_to_df(df):
    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
    def calculate(group):
        group['alpha84'] = np.where(group['close'].diff(1)>0,group['volume'],np.where(group['close'].diff(1)==0,0,-group['volume']))
        return group
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)


def Alpha080_to_df(df):
    # (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    def calculate(group):
        group['alpha80'] = (group['volume'] / (group['volume'].shift(5)+1e-10) -1) *100
        return group
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha071_to_df(df):
    # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    def calculate(group):
        group['alpha71'] = (group['close']/(group['close'].rolling(window=24,min_periods=24).mean()+1e-10) -1)*100
        return group
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha070_to_df(df):
    # STD(AMOUNT,6)
    def calculate(group):
        group['alpha70'] = group['turnover'].rolling(window=6,min_periods=6).std()
        return group
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha069_to_df(df):
    # (SUM(DTM,20)>SUM(DBM,20)？(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)：(SUM(DTM,20)=SUM(DBM,20)？0：(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
    # DTM = (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
    # DBM = (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    def calculate(group):
        group['DTM'] = np.where(group['open']<=group['open'].shift(1) , 0 , np.maximum(group['high']-group['open'] , group['open']-group['open'].shift(1)))
        group['DBM'] = np.where(group['open']>=group['open'].shift(1) , 0 , np.maximum(group['open']-group['low'] , group['open']-group['open'].shift(1)))
        condition1 = group['DTM'].rolling(window = 20).sum() > group['DBM'].rolling(window = 20).sum()
        condition2 = group['DTM'].rolling(window = 20).sum() < group['DBM'].rolling(window = 20).sum()
        group['alpha69'] = 0
        group['alpha69'][condition1] = (1 - group['DBM'].rolling(window = 20).sum()) / ((group['DTM'].rolling(window = 20).sum())[condition1] + 1e-10)
        group['alpha69'][condition2] = (group['DTM'].rolling(window = 20).sum()) / ((group['DBM'].rolling(window = 20).sum() - 1)[condition2] + 1e-10)
        columns = group.columns.difference(['DTM', 'DBM'])
        return group[columns]
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha060_to_df(df):
    def calculate(group):
        group['alpha60'] = ((group['close'] - group['low']) - (group['high'] - group['close'])) / (group['high'] - group['low'] + 1e-10) * group['volume']
        group['alpha60'] = group['alpha60'].rolling(window=20,min_periods=20).sum()
        return group
    df = df.groupby('symbol').apply(calculate)
    df = df.reset_index(drop=True)
    return df

def Alpha059_to_df(df):
    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
    def calculate(group):
        group['alpha59'] = 0
        condition1 = (group['close'].diff(1)>0)
        condition2 = (group['close'].diff(1)<0)
        
        group['alpha59'][condition1] = group['close'][condition1] - np.minimum(group['low'][condition1], group['close'].shift(1)[condition1])
        group['alpha59'][condition2] = group['close'][condition2] - np.maximum(group['high'][condition2], group['close'].shift(1)[condition2])
        group['alpha59'] = group['alpha59'].rolling(window = 20, min_periods=20).sum()
        return group
    df = df.groupby('symbol').apply(calculate) 
    return df.reset_index(drop=True)
    
def Alpha046_to_df(df):
    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    def calculate(group):
        group['alpha46'] = (group['close'].rolling(window=3).mean() + group['close'].rolling(window=6).mean() + group['close'].rolling(window=12).mean() + group['close'].rolling(window=24).mean()) /(4*group['close']+1e-10)
        return group
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha040_to_df(df):
    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
    def calculate(group):
        group['delay_close'] = group['close'].shift(1)
        group['alpha1'] = np.where(group['delay_close'] < group['close'], group['volume'], 0)
        group['alpha2'] = np.where(group['delay_close'] >= group['close'], group['volume'], 0)
        group['alpha40'] = group['alpha1'].rolling(window = 26,).sum() / (group['alpha2'].rolling(window = 26).sum()+1e-10) * 100
        columns = group.columns.difference(['delay_close', 'alpha1' , 'alpha2'])
        return group[columns]
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha031_to_df(df):
    # (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    def calculate(group):
        group['average'] = group['close'].rolling(window = 12, min_periods=12).mean() 
        group['alpha31'] = (group['close']-group['average'])/(group['average']+1e-10) * 100
        columns = group.columns.difference(['average'])
        return group[columns]
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha029_to_df(df):
    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    def calculate(group):
        group['alpha29'] = (group['close']-group['close'].shift(6))/(group['close'].shift(6)+1e-10)*group['volume']
        return group
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha027_to_df(df):
    # WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    def calculate(group):
        group['ret'] = (group['close']-group['close'].shift(3))/(group['close'].shift(3)+1e-10) * 100 + (group['close']-group['close'].shift(6))/(group['close'].shift(6)+1e-10) * 100
        weights = np.array([0.9**i for i in range(12)])[::-1]
        group['alpha27'] = group['ret'].rolling(window=12 , min_periods=12).apply(lambda x: np.dot(x,weights) / weights.sum())
        columns = group.columns.difference(['ret'])
        return group[columns]
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha026_to_df(df):
    # ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))

    df['vwap'] = df['turnover'] / (df['volume']+1e-10)
    
    # def calculate(group):
    #     group['avg_close_7'] = group['close'].rolling(window=7, min_periods=1).mean()
    #     group['close_diff'] = group['avg_close_7'] - group['close']
    #     group['corr_vwap_close'] = group['vwap'].rolling(window=230, min_periods=1).corr(group['close'].shift(5))
    #     group['alpha26'] = group['close_diff'] + group['corr_vwap_close']
    #     columns = group.columns.difference(['avg_close_7', 'close_diff' , 'corr_vwap_close'])
    #     return group[columns]
    # df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

def Alpha011_to_df(df):
    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    df['alpha11'] = (2*df['close']-df['low']-df['high'])/(df['high']-df['low']+1e-10)*df['volume']
    def calculate(group):
        group['alpha11'] = group['alpha11'].rolling(window=6 , min_periods=6).sum()
        return group
    df = df.groupby('symbol').apply(calculate)
    return df.reset_index(drop=True)

