import numpy as np
import pandas as pd

results = pd.read_csv('Datasets/results.csv')
print("load Results...")
print(results.head())

## filter year
year = []
for row in results['date']:
    year.append(int(row[:4]))
results['match_year'] = year
results = results[results['match_year'] >= 2014]
print("Filter year: 2014+")

## Categoryin goal difference
results['goal_difference'] = np.absolute(results['home_score'] - results['away_score'])
gd_category = []
for i in range(len(results['goal_difference'])):
    if results['goal_difference'].iloc[i] <= 0:
        gd_category.append('low')
    else:
        gd_category.append('high')
results['gd_category'] = gd_category
print("Calculate Goal Difference: ")
print(results.head())

winners = []
for i in range(len(results['home_team'])):
    if results['home_score'].iloc[i] > results['away_score'].iloc[i] and results['gd_category'].iloc[i] == 'high':
        winners.append(results['home_team'].iloc[i])
    elif results['home_score'].iloc[i] < results['away_score'].iloc[i] and results['gd_category'].iloc[i] == 'high':
        winners.append(results['away_team'].iloc[i])
    else:
        winners.append('Draw')
results['winning_team'] = winners
print("Get Winning Team: ")
print(results.head())

worldcup_teams = [
    'Australia', ' Iran', 'Japan', 'Korea Republic', 
    'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria', 
    'Senegal', 'Tunisia', 'Costa Rica', 'Mexico', 
    'Panama', 'Argentina', 'Brazil', 'Colombia', 
    'Peru', 'Uruguay', 'Belgium', 'Croatia', 
    'Denmark', 'England', 'France', 'Germany', 
    'Iceland', 'Poland', 'Portugal', 'Russia', 
    'Serbia', 'Spain', 'Sweden', 'Switzerland'
    ]

df_teams_home = results[results['home_team'].isin(worldcup_teams)]
df_teams_away = results[results['away_team'].isin(worldcup_teams)]
df_teams = pd.concat((df_teams_home, df_teams_away))
df_teams.drop_duplicates()
# print(df_teams.count())
print("Filter Team thats participate in World Cup 2018")
print(df_teams.head())

## drop kolom yang tidak mempengaruhi hasil match
df_teams = df_teams.drop(
    [
        'date',
        'home_score',
        'away_score',
        'tournament',
        'city',
        'country',
        'goal_difference',
        'match_year',
        'gd_category'
    ],
    axis=1
)
print("Drop Columns that doesnt affect match outcomes")
print(df_teams.head())

## BUILDING THE MODEL 
## Prediction target:
## "2" if home team wins, "1" if Draw, "0" if away team wins
df_teams = df_teams.reset_index(drop=True)
df_teams.loc[df_teams.winning_team == df_teams.home_team, 'winning_team'] = 2
df_teams.loc[df_teams.winning_team == 'Draw', 'winning_team'] = 1
df_teams.loc[df_teams.winning_team == df_teams.away_team, 'winning_team'] = 0

print(df_teams.head())

final = pd.get_dummies(
    df_teams,
    prefix=['home_team', 'away_team'],
    columns=['home_team', 'away_team']
    )

final.to_csv('final.csv')

print(final.head())
