import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import re


#First, we import the three schemes respectively:
#import first sheet: SDG Goals
sdg = pd.read_excel('sdg.xlsx', header=None, encoding='latin', skip_blank_lines=False)
sdg.fillna(0, inplace=True)
sdg = sdg.where(sdg != "x", 1)
sdg = sdg.where(sdg != "X", 1)


#import second sheet: SDG Targets
target = pd.read_excel('Target.xlsx', header=None, encoding='latin', skip_blank_lines=False)
target.fillna(0, inplace=True)
target = target.where(target != "x", 1)
target = target.where(target != "X", 1)

#import third sheet: SDG Indicators
indicator = pd.read_excel('Indicator.xlsx', header=None, encoding='latin', skip_blank_lines=False)
indicator.fillna(0, inplace=True)
indicator = indicator.where(indicator != "x", 1)
indicator = indicator.where(indicator != "X", 1)

# SDG Goals and keywords: return and convert arrays to proper strings
import sys
reload(sys)
sys.setdefaultencoding('utf8')

keys = sdg.iloc[0,2:1317]
keys = keys.astype(str)
keys = keys.reset_index()
keys = keys.drop(['index'], axis=1)

#strip strings
b = keys[0].tolist()
a = list(map(lambda x: x.replace(' to ',' ').replace(' at ',' ').replace(' of ',' ').replace(' for ',' ')
             .replace(' on ',' ').replace(' and/or ', ' ').replace(' or ', ' ').replace(' and ',' ').replace(' the ',' ').replace(' The ',' ')
             .replace(' their ',' ').replace(' from ',' ').replace(' in ',' ').replace(' all ',' ')
             .replace(' all ',' ').replace('\xe2\x80\x93', '-'),b))


strip = pd.DataFrame({0:a})

#because indicator has one key word more...
keys2 = indicator.iloc[0,4:1319]
keys2 = keys2.astype(str)
keys2 = keys2.reset_index()
keys2 = keys2.drop(['index'], axis=1)

#strip strings
b = keys2[0].tolist()
a = list(map(lambda x: x.replace(' to ',' ').replace(' at ',' ').replace(' of ',' ').replace(' for ',' ')
             .replace(' on ',' ').replace(' and/or ', ' ').replace(' or ', ' ').replace(' and ',' ').replace(' the ',' ').replace(' The ',' ')
             .replace(' their ',' ').replace(' from ',' ').replace(' in ',' ').replace(' all ',' ')
             .replace(' all ',' ').replace('\xe2\x80\x93', '-'),b))
strip2 = pd.DataFrame({0:a})

sdg.columns = sdg.iloc[0]
goals = sdg.iloc[1:18,0:2]
goals = goals.astype(str)
goals = goals.reset_index()
goals = goals.drop(['index'], axis=1)

#save original arrays to pickles. (just to be safe)
# import pickle
# pickle_out = open("dict.pickleA","wb")
# pickle.dump(goals, pickle_out)
# pickle_out.close()
#
# pickle_out = open("dict.pickleB","wb")
# pickle.dump(keys, pickle_out)
# pickle_out.close()

#apply mapping function to obtain binary decision matrix whether keyword is part of SDG Goals

goals.columns = ['Goal','Description']
res = []
# for a in goals['Description']:
#     res.append(keys.applymap(lambda x : " "+x.lower() in a.lower()))

# Let's define a mapping function over stripped strings that allows substrings to match as well to boost success rate
for a in goals['Description']:
    res.append(strip.applymap(lambda x: np.sum([i in a for i in x.split(' ')]) > 0))

sdg_mapped = pd.concat(res, axis=1).T
sdg_mapped.index = np.arange(len(sdg_mapped))
sdg_mapped = goals.join(sdg_mapped)
sdg_mapped.columns = sdg.iloc[0]
sdg = sdg.reindex(sdg.index.drop(0))

# SDG Target and keywords: return and convert arrays to proper strings
target.columns = target.iloc[0]
target_des = target.iloc[1:170,0:3]
target_des = target_des.astype(str)
target_des = target_des.reset_index()
target_des = target_des.drop(['index'], axis=1)

#apply mapping function to obtain binary decision matrix whether keyword is part of SDG Targets
target_des.columns = ['Goal','Target','Description']
res = []
# for a in target_des['Description']:
#     res.append(keys.applymap(lambda x : " "+x.lower() in a.lower()))

# Let's define a mapping function that allows substrings to match as well to boost success rate
for a in target_des['Description']:
    res.append(strip.applymap(lambda x: np.sum([i in a for i in x.split(' ')]) > 0))


target_mapped = pd.concat(res, axis=1).T
target_mapped.index = np.arange(len(target_mapped))
target_mapped = target_des.join(target_mapped)
target_mapped.columns = target.iloc[0]
target = target.reindex(target.index.drop(0))

# SDG Indicator and keywords: return and convert arrays to proper strings
indicator.columns = indicator.iloc[0]
indicator_des = indicator.iloc[1:246,0:4]
indicator_des = indicator_des.astype(str)
indicator_des = indicator_des.reset_index()
indicator_des = indicator_des.drop(['index'], axis=1)

#apply mapping function to obtain binary decision matrix whether keyword is part of SDG Indicators
indicator_des.columns = ['Goal', 'Target', 'Indicator', 'Description']
res = []
for a in indicator_des['Description']:
    res.append(strip2.applymap(lambda x : " "+x.lower() in a.lower()))

# for a in indicator_des['Description']:
#     res.append(keys2.applymap(lambda x: np.sum([i in a for i in x.split(' ')]) > 0))


indicator_mapped = pd.concat(res, axis=1).T
indicator_mapped.index = np.arange(len(indicator_mapped))
indicator_mapped = indicator_des.join(indicator_mapped)
indicator_mapped.columns = indicator.iloc[0]
indicator = indicator.reindex(indicator.index.drop(0))

#calculate Hit rate as share of True out of False
false = sdg_mapped[sdg_mapped==False].count()
true = sdg_mapped[sdg_mapped==True].count()
total = true.sum() + false.sum()
SDG_Goals_Hit_rate = true.sum()/float(total)
SDG_Goals_Hit_rate = str(round(SDG_Goals_Hit_rate, 4))
print "hit rate SDG_goals:", SDG_Goals_Hit_rate


false = target_mapped[target_mapped==False].count()
true = target_mapped[target_mapped==True].count()
total = true.sum() + false.sum()
SDG_Targets_Hit_rate = true.sum()/float(total)
SDG_Targets_Hit_rate = str(round(SDG_Targets_Hit_rate, 4))
print "hit rate SDG_Targets:", SDG_Targets_Hit_rate

false = indicator_mapped[indicator_mapped==False].count()
true = indicator_mapped[indicator_mapped==True].count()
total = true.sum() + false.sum()
SDG_Indicator_Hit_rate = true.sum()/float(total)
SDG_Indicator_Hit_rate = str(round(SDG_Indicator_Hit_rate, 4))
print "hit rate SDG_Indicators:", SDG_Indicator_Hit_rate


#compare mapping function with excel entries and identify most mentioned goal/target/indicator/overall according to keywords
#GOALS
#sdg_mapped['sum_python'] = sdg_mapped.sum(axis=1)
sdg_mapped['sum_python'] = sdg_mapped.iloc[:, 2:].sum(axis=1)
compare1 = sdg_mapped[['Goal','Description','sum_python']].copy()
del sdg['Goal']
del sdg['Description']
sdg['sum_excel'] = sdg.sum(axis=1)
compare2 = sdg['sum_excel'].copy()
compare2 = compare2.reset_index()
compare2 = compare2.drop(['index'], axis=1)
frames = [compare1, compare2]
goals_total = pd.concat(frames, axis=1)
#print "Mapping Function for Goals/Targets and Indicators shows more appearances than excel sheet indicated"

#Targets
target_mapped['sum_python'] = target_mapped.iloc[:, 4:].sum(axis=1)
compare3 = target_mapped[['Goal', 'Target','Description','sum_python']].copy()
del target['Goal']
del target['Description']
del target['Target']
target['sum_excel'] = target.sum(axis=1)
compare4 = target['sum_excel'].copy()
compare4 = compare4.reset_index()
compare4 = compare4.drop(['index'], axis=1)
frames = [compare3, compare4]
targets_total = pd.concat(frames, axis=1)


#Indicators
indicator_mapped['sum_python'] = indicator_mapped.iloc[:, 5:].sum(axis=1)
compare5 = indicator_mapped[['Goal', 'Target', 'Indicator', 'Description','sum_python']].copy()
del indicator['Goal']
del indicator['Target']
del indicator['Indicator']
del indicator['Description']
indicator['sum_excel'] = indicator.sum(axis=1)
compare6 = indicator['sum_excel'].copy()
compare6 = compare6.reset_index()
compare6 = compare6.drop(['index'], axis=1)
frames = [compare5, compare6]
indicators_total = pd.concat(frames, axis=1)

#Summary Matches and Comparison to Excel Sheet
# print "Mapping Function for Goals/Targets/Indicators shows overall more appearances than excel sheet indicates"
# print "Top 3 Goals with keywords Matches: SDG 8 (11 matches), 14 (10 matches), 15 (10 matches)"
# print "Top 5 individual Targets with keywords Matches: Target 1.4 (25 matches), 2.3 (31 ma.), 2.4 (25 ma., 6.a (23 ma.), 11.b (27 ma.)"
# print "Top indivual Indicators: 1.3.1 (19 ma.), 3.81. (18 ma.), 4.7.1 (21 ma.), 7.b.1 (18 ma.), 11.4.1 (18 ma.), 12.8.1 (19 ma.), 13.b.1 (20 ma.)R"
#
# print "Goal 1 has 1 match for itself, 85 matches for its targets and 115 matches for its indicators. How weight? Weight at all? I need df that assigns goals properly!"
# print "If Goal 1 has accumulated 201 matches, is it more or less worth than 300 matches for goal 2 with differen keywords, do I have to weight keywords?"

#Matches conditional on goals:
sdg_mapped['Goal'] = sdg_mapped['Goal'].astype(float)
target_mapped['Goal'] = target_mapped['Goal'].astype(float)
indicator_mapped['Goal'] = indicator_mapped['Goal'].astype(float)
target_mapped['sum_by_goal'] = target_mapped['Goal'].apply(lambda x:  target_mapped[target_mapped['Goal'] == x]['sum_python'].sum())
indicator_mapped['sum_by_goal2'] = indicator_mapped['Goal'].apply(lambda x:  indicator_mapped[indicator_mapped['Goal'] == x]['sum_python'].sum())

#Top Keywords Goals/targets/indicator Add row that show sums of each column.
sum = sdg_mapped.iloc[:, 3:-1].sum()
sdg_mapped = sdg_mapped.append(sum, ignore_index=True)
sdg_mapped= sdg_mapped.replace('nan', 0).fillna(0)
print "Top keywords Goals:"
print sdg_mapped.T[17].sort_values(ascending=False).iloc[:10]

sum = target_mapped.iloc[:, 4:-2].sum()
target_mapped = target_mapped.append(sum, ignore_index=True)
target_mapped= target_mapped.replace('nan', 0).fillna(0)
print "Top keywords Targets:"
print target_mapped.T[169].sort_values(ascending=False).iloc[:10]

sum = indicator_mapped.iloc[:, 5:-2].sum()
indicator_mapped = indicator_mapped.append(sum, ignore_index=True)
indicator_mapped= indicator_mapped.replace('nan', 0).fillna(0)
print "Top keywords Indicators:"
print indicator_mapped.T[244].sort_values(ascending=False).iloc[:10]

# print "One problem is that we have to define what 'keyword in Goal/Indicator/Target' really means. " \
#       "For instance,  I had many 'men' being part of 'government'. However it is obviously not about men."

#Overall top matches Goals+targets*Indicator
sum_goals = sdg_mapped[['Goal','sum_python']]
sum_targets = target_mapped[['Goal','sum_by_goal']]
sum_indicators = indicator_mapped[['Goal','sum_by_goal2']]
sum_goals = sum_goals.drop(sum_goals.index[17])
sum_targets = sum_targets.drop(sum_targets.index[169])
sum_indicators = sum_indicators.drop(sum_indicators.index[244])

overall = sum_goals.merge(sum_targets, how="left")
overall = overall.merge(sum_indicators, how="left")
overall = overall[~overall.Goal.duplicated(keep='first')]
overall = overall.reset_index()
overall = overall.drop(['index'], axis=1)

overall['total'] = overall.iloc[:, 1:].sum(axis=1)

print "total matches per Goal chronologically"
print overall['total']

#Overall key words Goals+targets+indicators
key_goals = sdg_mapped.iloc[17:]
key_targets = target_mapped.iloc[169:]
key_indicators = indicator_mapped.iloc[244:]

overall_keys = key_goals.append(key_targets)
overall_keys = overall_keys.append(key_indicators)
overall_keys = overall_keys.reset_index()
overall_keys = overall_keys.drop(['index'], axis=1)

sum = overall_keys.iloc[:,:].sum()
overall_keys = overall_keys.append(sum, ignore_index=True)
print "Top Overall Keywords:"
print overall_keys.T[3].sort_values(ascending=False).iloc[:15]








