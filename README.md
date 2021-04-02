# Agitation Dataset from 2020 Polish Presidential Elections
The popularity of social media makes politicians use it for political campaigns. 
Therefore, social media are full of political ads posted by candidates and other 
messages that influence the electorate. From the election point of view, all 
defamatory or false content that influences the voters is illegal. It can be 
subjected to an expedited court proceeding by competing, affected candidates but
the real problems rest in a burden of proof. The election administration cannot 
track the spread and quantity of messages that count as political agitation 
(electioneering) under the election code. Hence, we present the first publicly 
open dataset for detecting political agitation in the Polish language. It contains
6,112 human-annotated tweets tagged with four legally conditioned categories. 
We achieved a 0.66 inter-annotator agreement (Cohen’s kappa score). An additional
annotator resolved the mismatches between the first two improving the annotation
process's consistency and complexity. We used the newly created dataset to train 
a HerBERT model (achieving a 68% F1 score). We also present a possible direction
of use cases for such datasets and models, enriching the paper with an analysis 
of the Polish 2020 Presidential Elections on Twitter.

## Interactive dashboards and poster
The outcome of our work is also available at [www.smart-wust.ml](http://www.smart-wust.ml/) in the section
*Agitation*, where one can reach additional analysis and insights as well as examples of model's
usage including the scoring of own entered text. You can also see the result of our work 
in the form of a [poster](assets/Is_agitation_flooding_social_media.pdf).

## Requirements
Python >= 3.8

## Installation
Clone the repository and then in the project folder run
```bash
pip install -r requirements.txt
```

## Example of model usage
We trained a classifier based on a transformer language model HerBERT which is dedicated 
to the Polish language. Finally, we achieved 68% F1 score for the stratified train-test 
split in an 80/20 ratio.

Open python or jupyter notebook and run

```python
from electoral_agitation.model import AgitationModel

model = AgitationModel.from_pretrained()
model.predict([
    "Błagam zagłosujcie na Trzaskowskiego w drugiej turze!",
    "Uwielbiam Andrzeja Dudę, tam gdzie się pojawia oglądalność bije rekordy.",
    "Zagłosuj w tych wyborach! To ważny dzień.",
    "Tesco zmienia sposób pakowania kurczaków.",
])
# ['inducement' 'encouragement' 'voting_turnout' 'normal']
```

## Dataset loading
```python
import pandas as pd
df = pd.read_json("data/electoral_agitation_dataset.jsonl", orient='records', lines=True, dtype='int64')
```

### ATTENTION: Raw text of tweets is not available
The dataset file does not contain the text of tweets due to the 
[Twitter policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy). 
It contains tweet ID, URL, label and meta-data assigned to the tweet.
![DATASET_SAMPLE](/assets/dataset_sample.png)

**If you want to get tweet text**, you should either download them yourself via Twitter API 
(using tweet ids or their URLs) or **contact us, and we will share the whole dataset with 
our collaborators (much more relaxed and faster solution).**  

## Citing

## Acknowledgment
