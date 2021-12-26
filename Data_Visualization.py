#import matplotlib
import json
import numpy as np
import matplotlib.pyplot as plt


# Visualization of the class distribution of objects over all tfrecord files
with open('./Training_ClassComp_Normalized.json') as f:
    predictions = json.load(f)

d2 = {}
filenames = []
vehicles = []
pedestrians = []
cyclists = []
labels = []
for record in predictions:
    filenames.append(record)
    labels.append(record.split('segment-')[-1][:10])
    vehicles.append(predictions[record]["1"])
    pedestrians.append(predictions[record]["2"])
    cyclists.append(predictions[record]["4"])
    # d2[record] = {"1": round(predictions[record]["1"]/10000,3)}
    # d2[record]["2"] = round(predictions[record]["2"]/10000,3)
    # d2[record]["4"] = round(predictions[record]["4"]/10000,3)

# with open('./Training_ClassComp_Normalized.json', 'w') as f:
#     print(d2,file=f)

#num_ojbects = sum(vehicles) + sum(pedestrians) + sum(cyclists)
vehicles = np.array(vehicles)
pedestrians = np.array(pedestrians)
cyclists = np.array(cyclists)

width = 0.8       # the width of the bars

fig, ax = plt.subplots()

ax.bar(labels, vehicles, width, label='vehicles',align='center')
ax.bar(labels, pedestrians, width, bottom=vehicles, label='pedestrians',align='center')
ax.bar(labels, cyclists, width,  bottom=vehicles+pedestrians, label='cyclists',align='center')

ax.set_ylabel('Average number of objects per image')
ax.set_title('Average distribution of objects per image over tfrecords')
plt.xticks(rotation='vertical')
ax.legend()

plt.show()



with open('Training_WeatherCondition_FilesOnly.json') as f:
    weather_cond = json.load(f)

image_class = weather_cond['Classification']
labels = []
condition_count = []

for weather in image_class:
    labels.append(weather)
    condition_count.append(len(image_class[weather]))

print(condition_count, labels)
print(list(range(1,len(labels))))
plt.bar(list(range(1,len(labels)+1)), height=condition_count)
plt.xticks(list(range(1,len(labels)+1)), labels)
plt.ylabel('Number of records')
plt.title('Number of records with certain weather conditions (a total of97 train/val tfrecords was analysed)')

plt.show()