# Blood glucose Forecasting Approaches

This repository contains simple approaches to train blood glucose forecasting models given the patient's past. 
Models are trained and validated on the  [OhioT1DM dataset](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) (2018 and 2020). 
All approaches are visible in various Jupyter notebooks.



## Requirements

The "Ohio Data/" folder must be in the repository root directory with the following structure: 

```
project/
|
|...
|
|--Ohio Data/
   |
   |--Ohio2018/
   |  |
   |  |--test/
   |  |  |
   |  |  |{patient_id}-ws-testing_processed.csv
   |  |
   |  |--train/
   |     |
   |     |{patient_id}-ws-training_processed.csv
   |
   |--Ohio2018/
   |  |
   |  |--test/
   |  |  |
   |  |  |{patient_id}-ws-testing_processed.csv
   |  |
   |  |--train/
   |     |
   |     |{patient_id}-ws-training_processed.csv    

```

```
# Python 3.8 or higher

pip3 install -r requirements.txt

jupyter-lab
`````



The following sections describe my approach.

# Research

## Prior Knowledge

Most of my experience is in the field of computer vision. 
When it comes to tasks related to time series, I only have experience with anomaly detection.
I have rarely used sequential neural networks, such as RNNs or LSTMs.
In contrast, I was able to gain a lot of experience with CNNs.

## Pytorch Forecasting

I found this framework just by coincidence. 
It seems to be the equivalent of what fastAI is for Pytorch (or even Keras is for TensorFlow).
Besides forecasting models, the framework implements various features for data preparation.

For example: 

* Temporal time encoding
* Sampling of missing values
* Dataloader generator

## (Blood glucose) Forecasting Papers

Unfortunately, as I am not a student anymore, I am unable to read some papers without being charged. 
Therefore I am limited to free paper I can find on the internet. 
Nevertheless, here is a list of papers I read or at least took a look at for preparation: 

1. [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363)
   * A derivation of the classical transformer model, specifically designed for time series forecasting
   * Distinguishes between categorical and continuous data
   * Can also take future **known** values as input
   * Achieves state-of-the-art results in forecasting tasks
2. [N-Hits](https://arxiv.org/pdf/2201.12886.pdf)
   * An enhanced version of N-BEATS
   * Instead of N-BEATS, N-Hits predicts interpolation coefficients to interpolate values across a time series
   * Also utilizes average pooling layers per block for 
3. [Using N-BEAT to forecast blood glucose values](http://ceur-ws.org/Vol-2675/paper18.pdf)
   * Uses a customized N-BEATS model to predict blood glucose values
   * The major difference is to include an LSTM inside the blocks
   * Also uses a customized loss function
4. [Using GANS](https://discovery.ucl.ac.uk/id/eprint/10115176/1/paper15.pdf)
   * The generator generates the future blood glucose values up to a defined prediction horizon
   * The discriminator discriminates between ground truth and generated blood glucose values
   * My opinion: 
     * Even though the results from the authors look promising, I can not imagine that this approach can beat other models (LSTM, Transformer, ...)
     * In the past, I experienced how hard it can be to train GANs
     * Furthermore, they require a lot of computational resources
5. [Comparison of different methods for blood glucose prediction](https://arxiv.org/pdf/2109.02178.pdf)
   * A comparison between many approaches for blood glucose forecasting
   * An LSTM Ensemble model achieved the best results 

## Reinforcement Learning

As the task states: deep reinforcement learning (DRL) can be used to solve this task. 
Intuitively this does not make much sense, as DRL is usually used to maximize a future outcome instead of just predicting future values.
Many papers regarding stock trading describe their approach of using DLR techniques by taking past and current time series of stock prices. 
The major difference between trading strategies and general time series forecasting is that trading strategies aim to maximize their future portfolio value instead of just predicting the stock prices. 
Therefore it makes sense to use DRL for this task. 

In this task, on the other hand, it would make sense if you would, e.g. measure what happens if you take treatment. 

Furthermore, I could not find a single paper regarding blood glucose forecasting using DRL methods. 
I rarely find any prior time series forecasting work using reinforcement learning in general. 
Therefore I will only use "traditional" methods to solve this task. 


### Prior Knowledge

* DRL: 
  * Unfortunately I have close to zero prior knowledge about DRL
  * I know some basic terms (MDP, reward, Bellman Equation, return, ...)
  * I tried out Deep Q Learning for simple tasks 
  * I read the [paper of Alpha Zero](https://arxiv.org/abs/1712.01815) out of curiosity. I understood the basic ideas but never reimplemented it on my own
  * I also read the [paper of ReBel](https://arxiv.org/abs/2007.13544) in the past


# Approach

## Data preparation

First of all, I took a deeper look into the data set. 
My data analysis is viewable in the "data.ipynb" notebook.

### Summary

* The data only contains continuous data, 5-minute timestamps, and no(!) categorical data
* Many values (from target and non-target columns) are missing
  * There are extremely sparse columns like "carbInput" (98% missing)
  * But I believe that they can still contribute to the forecasting (e.g. after carbInput -> blood glucose should increase)
  * Except for the target column (cbg), missing values are interpolated by cubic splines
    * (One can argue that it doesn't make sense to interpolate, e.g. carbInput, because how would you interpolate if someone just ate?)
  * Because models work better with values between 0 and 1, all values are scaled accordingly (divided by max values of train set)
* Correlations: 
  * I scatter plotted the relations between every column and the respective cbg values
  * I noticed that the finger value is not as accurate as I expected (I expected an almost straight line)
  * Outliers: 
    * There are only a few clear outliers (one carbInput, some hr)
    * Unfortunately, after looking at the points in time where they appear, I could not determine why this is the case
    * One can argue that removing them would be reasonable, but I decided to include them
* Added data: 
  * Some models perform better when taking temporal information as input
  * Therefore I added positional embedded information, using sin/cos embedding
* Removed data:
  * If a data pair (input, label) contains at least one point with at least one missing cbg value, it is removed from the dataset

# Metrics

I focussed on metrics other researchers used to evaluate their models (rMSE and MAE).
The models I used took as input 24 past time steps (12hours) and a prediction horizon of 6/12 time steps (30min/60min).

## Models

For each model, there is a separate notebook where I explain my approaches ({model_name}_approach.ipynb).

Models:
1. N-BEATS (plain)
   * 12 blocks
   * loss function as described in [this paper](http://ceur-ws.org/Vol-2675/paper18.pdf)
2. N-BEATS (paper) (from [paper](http://ceur-ws.org/Vol-2675/paper18.pdf))
   * 12 blocks
   * loss function as described in [this paper](http://ceur-ws.org/Vol-2675/paper18.pdf)
3. LSTM
   * teacher enforecement is enabled for training
   * non bidirectional
   * single layer
4. LSTM
   * teacher enforecement is enabled for training
   * bidirectional
   * two layers
4. Ensemble (LSTM, N-BEATS)

# Results

## N-Beats (plain)

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-uog8" colspan="4">Prediction Horizon</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"></td>
    <td class="tg-0lax" colspan="2">30 Minutes</td>
    <td class="tg-0lax" colspan="2">60 Minutes</td>
  </tr>
  <tr>
    <td class="tg-0lax">Participant ID</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
  </tr>
  <tr>
    <td class="tg-0lax">559</td>
    <td class="tg-0lax">28.83</td>
    <td class="tg-0lax">20.84</td>
    <td class="tg-0lax">39.</td>
    <td class="tg-0lax">29.27</td>
  </tr>
  <tr>
    <td class="tg-0lax">563</td>
    <td class="tg-0lax">24.96</td>
    <td class="tg-0lax">18.64</td>
    <td class="tg-0lax">33.40</td>
    <td class="tg-0lax">25.22</td>
  </tr>
  <tr>
    <td class="tg-0lax">570</td>
    <td class="tg-0lax">24.20</td>
    <td class="tg-0lax">18.52</td>
    <td class="tg-0lax">34.15</td>
    <td class="tg-0lax">26.53</td>
  </tr>
  <tr>
    <td class="tg-0lax">575</td>
    <td class="tg-0lax">26.90</td>
    <td class="tg-0lax">19.57</td>
    <td class="tg-0lax">35.62</td>
    <td class="tg-0lax">26.76</td>
  </tr>
  <tr>
    <td class="tg-0lax">588</td>
    <td class="tg-0lax">26.03</td>
    <td class="tg-0lax">19.03</td>
    <td class="tg-0lax">34.52</td>
    <td class="tg-0lax">25.39</td>
  </tr>
  <tr>
    <td class="tg-0lax">591</td>
    <td class="tg-0lax">25.76</td>
    <td class="tg-0lax">19.76</td>
    <td class="tg-0lax">34.26</td>
    <td class="tg-0lax">26.80</td>
  </tr>
  <tr>
    <td class="tg-0lax">540</td>
    <td class="tg-0lax">34.74</td>
    <td class="tg-0lax">26.04</td>
    <td class="tg-0lax">43.78</td>
    <td class="tg-0lax">32.84</td>
  </tr>
  <tr>
    <td class="tg-0lax">544</td>
    <td class="tg-0lax">24.36</td>
    <td class="tg-0lax">18.58</td>
    <td class="tg-0lax">34.62</td>
    <td class="tg-0lax">27.24</td>
  </tr>
  <tr>
    <td class="tg-0lax">552</td>
    <td class="tg-0lax">25.24</td>
    <td class="tg-0lax">18.59</td>
    <td class="tg-0lax">32.35</td>
    <td class="tg-0lax">24.79</td>
  </tr>
  <tr>
    <td class="tg-0lax">567</td>
    <td class="tg-0lax">31.78</td>
    <td class="tg-0lax">23.35</td>
    <td class="tg-0lax">41.11</td>
    <td class="tg-0lax">31.40</td>
  </tr>
  <tr>
    <td class="tg-0lax">584</td>
    <td class="tg-0lax">30.27</td>
    <td class="tg-0lax">22.77</td>
    <td class="tg-0lax">39.93</td>
    <td class="tg-0lax">30.68</td>
  </tr>
  <tr>
    <td class="tg-0lax">596</td>
    <td class="tg-0lax">24.59</td>
    <td class="tg-0lax">18.34</td>
    <td class="tg-0lax">34.46</td>
    <td class="tg-0lax">26.16</td>
  </tr>
  <tr>
    <td class="tg-0lax">mean</td>
    <td class="tg-0lax">20.34</td>
    <td class="tg-0lax">27.50</td>
    <td class="tg-0lax">36.67</td>
    <td class="tg-0lax">27.76</td>
  </tr>
</tbody>
</table>

## N-Beats advanced

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-uog8" colspan="4">Prediction Horizon</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"></td>
    <td class="tg-0lax" colspan="2">30 Minutes</td>
    <td class="tg-0lax" colspan="2">60 Minutes</td>
  </tr>
  <tr>
    <td class="tg-0lax">Participant ID</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
  </tr>
  <tr>
    <td class="tg-0lax">559</td>
    <td class="tg-0lax">43.75</td>
    <td class="tg-0lax">32.76</td>
    <td class="tg-0lax">54.67</td>
    <td class="tg-0lax">41.03</td>
  </tr>
  <tr>
    <td class="tg-0lax">563</td>
    <td class="tg-0lax">32.04</td>
    <td class="tg-0lax">24.45</td>
    <td class="tg-0lax">38.03</td>
    <td class="tg-0lax">29.83</td>
  </tr>
  <tr>
    <td class="tg-0lax">570</td>
    <td class="tg-0lax">38.59</td>
    <td class="tg-0lax">31.39</td>
    <td class="tg-0lax">54.13</td>
    <td class="tg-0lax">45.11</td>
  </tr>
  <tr>
    <td class="tg-0lax">575</td>
    <td class="tg-0lax">37.1</td>
    <td class="tg-0lax">29.01</td>
    <td class="tg-0lax">44.85</td>
    <td class="tg-0lax">36.32</td>
  </tr>
  <tr>
    <td class="tg-0lax">588</td>
    <td class="tg-0lax">34.23</td>
    <td class="tg-0lax">25.48</td>
    <td class="tg-0lax">40.60</td>
    <td class="tg-0lax">30.91</td>
  </tr>
  <tr>
    <td class="tg-0lax">591</td>
    <td class="tg-0lax">34.44</td>
    <td class="tg-0lax">27.93</td>
    <td class="tg-0lax">40.16</td>
    <td class="tg-0lax">32.94</td>
  </tr>
  <tr>
    <td class="tg-0lax">540</td>
    <td class="tg-0lax">45.19</td>
    <td class="tg-0lax">34.26</td>
    <td class="tg-0lax">51.76</td>
    <td class="tg-0lax">39.70</td>
  </tr>
  <tr>
    <td class="tg-0lax">544</td>
    <td class="tg-0lax">37.29</td>
    <td class="tg-0lax">31.17</td>
    <td class="tg-0lax">43.83</td>
    <td class="tg-0lax">36.83</td>
  </tr>
  <tr>
    <td class="tg-0lax">552</td>
    <td class="tg-0lax">34.60</td>
    <td class="tg-0lax">28.25</td>
    <td class="tg-0lax">40.61</td>
    <td class="tg-0lax">33.35</td>
  </tr>
  <tr>
    <td class="tg-0lax">567</td>
    <td class="tg-0lax">41.34</td>
    <td class="tg-0lax">32.99</td>
    <td class="tg-0lax">46.82</td>
    <td class="tg-0lax">38.01</td>
  </tr>
  <tr>
    <td class="tg-0lax">584</td>
    <td class="tg-0lax">40.29</td>
    <td class="tg-0lax">31.77</td>
    <td class="tg-0lax">48.77</td>
    <td class="tg-0lax">39.64</td>
  </tr>
  <tr>
    <td class="tg-0lax">596</td>
    <td class="tg-0lax">35.45</td>
    <td class="tg-0lax">27.35</td>
    <td class="tg-0lax">40.97</td>
    <td class="tg-0lax">32.42</td>
  </tr>
  <tr>
    <td class="tg-0lax">mean</td>
    <td class="tg-0lax">31.05</td>
    <td class="tg-0lax">39.11</td>
    <td class="tg-0lax">45.77</td>
    <td class="tg-0lax">36.34</td>
  </tr>
</tbody>
</table>

## Plain LSTM

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-uog8" colspan="4">Prediction Horizon</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"></td>
    <td class="tg-0lax" colspan="2">30 Minutes</td>
    <td class="tg-0lax" colspan="2">60 Minutes</td>
  </tr>
  <tr>
    <td class="tg-0lax">Participant ID</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
  </tr>
  <tr>
    <td class="tg-0lax">559</td>
    <td class="tg-0lax">14.88</td>
    <td class="tg-0lax">9.81</td>
    <td class="tg-0lax">35.17</td>
    <td class="tg-0lax">26.02</td>
  </tr>
  <tr>
    <td class="tg-0lax">563</td>
    <td class="tg-0lax">14.71</td>
    <td class="tg-0lax">9.96</td>
    <td class="tg-0lax">29.45</td>
    <td class="tg-0lax">22.01</td>
  </tr>
  <tr>
    <td class="tg-0lax">570</td>
    <td class="tg-0lax">12.18</td>
    <td class="tg-0lax">8.31</td>
    <td class="tg-0lax">28.45</td>
    <td class="tg-0lax">21.73</td>
  </tr>
  <tr>
    <td class="tg-0lax">575</td>
    <td class="tg-0lax">16.94</td>
    <td class="tg-0lax">10.51</td>
    <td class="tg-0lax">31.57</td>
    <td class="tg-0lax">23.57</td>
  </tr>
  <tr>
    <td class="tg-0lax">588</td>
    <td class="tg-0lax">14.60</td>
    <td class="tg-0lax">9.92</td>
    <td class="tg-0lax">30.13</td>
    <td class="tg-0lax">22.36</td>
  </tr>
  <tr>
    <td class="tg-0lax">591</td>
    <td class="tg-0lax">16.37</td>
    <td class="tg-0lax">11.10</td>
    <td class="tg-0lax">31.08</td>
    <td class="tg-0lax">23.72</td>
  </tr>
  <tr>
    <td class="tg-0lax">540</td>
    <td class="tg-0lax">18.36</td>
    <td class="tg-0lax">12.42</td>
    <td class="tg-0lax">38.44</td>
    <td class="tg-0lax">28.69</td>
  </tr> 
  <tr>
    <td class="tg-0lax">544</td>
    <td class="tg-0lax">13.97</td>
    <td class="tg-0lax">9.60</td>
    <td class="tg-0lax">31.11</td>
    <td class="tg-0lax">24.78</td>
  </tr>
  <tr>
    <td class="tg-0lax">552</td>
    <td class="tg-0lax">13.62</td>
    <td class="tg-0lax">9.11</td>
    <td class="tg-0lax">28.45</td>
    <td class="tg-0lax">21.85</td>
  </tr>
  <tr>
    <td class="tg-0lax">567</td>
    <td class="tg-0lax">17.86</td>
    <td class="tg-0lax">11.65</td>
    <td class="tg-0lax">36.71</td>
    <td class="tg-0lax">27.37</td>
  </tr>
  <tr>
    <td class="tg-0lax">584</td>
    <td class="tg-0lax">16.62</td>
    <td class="tg-0lax">11.31</td>
    <td class="tg-0lax">32.98</td>
    <td class="tg-0lax">25.02</td>
  </tr>
  <tr>
    <td class="tg-0lax">596</td>
    <td class="tg-0lax">13.98</td>
    <td class="tg-0lax">9.33</td>
    <td class="tg-0lax">29.42</td>
    <td class="tg-0lax">21.95</td>
  </tr>
  <tr>
    <td class="tg-0lax">mean</td>
    <td class="tg-0lax">15.44</td>
    <td class="tg-0lax">10.25</td>
    <td class="tg-0lax">32.07</td>
    <td class="tg-0lax">24.01</td>
  </tr>
</tbody>
</table>


## Multistacked Bidirectional LSTM 


<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-uog8" colspan="4">Prediction Horizon</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"></td>
    <td class="tg-0lax" colspan="2">30 Minutes</td>
    <td class="tg-0lax" colspan="2">60 Minutes</td>
  </tr>
  <tr>
    <td class="tg-0lax">Participant ID</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
  </tr>
  <tr>
    <td class="tg-0lax">559</td>
    <td class="tg-0lax">25.981</td>
    <td class="tg-0lax">19.40</td>
    <td class="tg-0lax">46.13</td>
    <td class="tg-0lax">34.09</td>
  </tr>
  <tr>
    <td class="tg-0lax">563</td>
    <td class="tg-0lax">23.52</td>
    <td class="tg-0lax">18.60</td>
    <td class="tg-0lax">35.48</td>
    <td class="tg-0lax">27.68</td>
  </tr>
  <tr>
    <td class="tg-0lax">570</td>
    <td class="tg-0lax">24.72</td>
    <td class="tg-0lax">20.73</td>
    <td class="tg-0lax">49.01</td>
    <td class="tg-0lax">39.66</td>
  </tr>
  <tr>
    <td class="tg-0lax">575</td>
    <td class="tg-0lax">24.89</td>
    <td class="tg-0lax">18.60</td>
    <td class="tg-0lax">37.92</td>
    <td class="tg-0lax">29.56</td>
  </tr>
  <tr>
    <td class="tg-0lax">588</td>
    <td class="tg-0lax">23.35</td>
    <td class="tg-0lax">17.93</td>
    <td class="tg-0lax">36.92</td>
    <td class="tg-0lax">29.35</td>
  </tr>
  <tr>
    <td class="tg-0lax">591</td>
    <td class="tg-0lax">22.99</td>
    <td class="tg-0lax">17.77</td>
    <td class="tg-0lax">45.64</td>
    <td class="tg-0lax">34.30</td>
  </tr>
  <tr>
    <td class="tg-0lax">540</td>
    <td class="tg-0lax">26.27</td>
    <td class="tg-0lax">19.62</td>
    <td class="tg-0lax">36.28</td>
    <td class="tg-0lax">28.08</td>
  </tr>
  <tr>
    <td class="tg-0lax">544</td>
    <td class="tg-0lax">21.29</td>
    <td class="tg-0lax">15.45</td>
    <td class="tg-0lax">35.97</td>
    <td class="tg-0lax">28.31</td>
  </tr>
  <tr>
    <td class="tg-0lax">552</td>
    <td class="tg-0lax">20.17</td>
    <td class="tg-0lax">14.93</td>
    <td class="tg-0lax">40.57</td>
    <td class="tg-0lax">31.21</td>
  </tr>
  <tr>
    <td class="tg-0lax">567</td>
    <td class="tg-0lax">25.13</td>
    <td class="tg-0lax">18.13</td>
    <td class="tg-0lax">40.57</td>
    <td class="tg-0lax">31.21</td>
  </tr>
  <tr>
    <td class="tg-0lax">584</td>
    <td class="tg-0lax">25.31</td>
    <td class="tg-0lax">18.75</td>
    <td class="tg-0lax">42.21</td>
    <td class="tg-0lax">31.90</td>
  </tr>
  <tr>
    <td class="tg-0lax">596</td>
    <td class="tg-0lax">19.98</td>
    <td class="tg-0lax">14.92</td>
    <td class="tg-0lax">35.98</td>
    <td class="tg-0lax">27.42</td>
  </tr>
  <tr>
    <td class="tg-0lax">mean</td>
    <td class="tg-0lax">23.72</td>
    <td class="tg-0lax">17.90</td>
    <td class="tg-0lax">40.40</td>
    <td class="tg-0lax">31.00</td>
  </tr>
</tbody>
</table>

## Ensemble Model (Plain LSTM and Plain N-BEATS)


<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-uog8" colspan="4">Prediction Horizon</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"></td>
    <td class="tg-0lax" colspan="2">30 Minutes</td>
    <td class="tg-0lax" colspan="2">60 Minutes</td>
  </tr>
  <tr>
    <td class="tg-0lax">Participant ID</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
    <td class="tg-0lax">rMSE</td>
    <td class="tg-0lax">MAE</td>
  </tr>
  <tr>
    <td class="tg-0lax">559</td>
    <td class="tg-0lax">14.23</td>
    <td class="tg-0lax">9.22</td>
    <td class="tg-0lax">26.52</td>
    <td class="tg-0lax">17.27</td>
  </tr>
  <tr>
    <td class="tg-0lax">563</td>
    <td class="tg-0lax">13.91</td>
    <td class="tg-0lax">9.00</td>
    <td class="tg-0lax">20.64</td>
    <td class="tg-0lax">13.92</td>
  </tr>
  <tr>
    <td class="tg-0lax">570</td>
    <td class="tg-0lax">12.27</td>
    <td class="tg-0lax">8.27</td>
    <td class="tg-0lax">21.87</td>
    <td class="tg-0lax">15.29</td>
  </tr>
  <tr>
    <td class="tg-0lax">575</td>
    <td class="tg-0lax">16.19</td>
    <td class="tg-0lax">9.88</td>
    <td class="tg-0lax">25.53</td>
    <td class="tg-0lax">17.53</td>
  </tr>
  <tr>
    <td class="tg-0lax">588</td>
    <td class="tg-0lax">13.79</td>
    <td class="tg-0lax">9.15</td>
    <td class="tg-0lax">21.08</td>
    <td class="tg-0lax">14.45</td>
  </tr>
  <tr>
    <td class="tg-0lax">591</td>
    <td class="tg-0lax">15.49</td>
    <td class="tg-0lax">10.22</td>
    <td class="tg-0lax">23.98</td>
    <td class="tg-0lax">16.81</td>
  </tr>
  <tr>
    <td class="tg-0lax">540</td>
    <td class="tg-0lax">16.13</td>
    <td class="tg-0lax">10.83</td>
    <td class="tg-0lax">28.51</td>
    <td class="tg-0lax">19.34</td>
  </tr>
  <tr>
    <td class="tg-0lax">544</td>
    <td class="tg-0lax">13.16</td>
    <td class="tg-0lax">8.90</td>
    <td class="tg-0lax">22.89</td>
    <td class="tg-0lax">16.43</td>
  </tr>
  <tr>
    <td class="tg-0lax">552</td>
    <td class="tg-0lax">12.16</td>
    <td class="tg-0lax">8.40</td>
    <td class="tg-0lax">21.09</td>
    <td class="tg-0lax">14.50</td>
  </tr>
  <tr>
    <td class="tg-0lax">567</td>
    <td class="tg-0lax">15.66</td>
    <td class="tg-0lax">10.30</td>
    <td class="tg-0lax">27.21</td>
    <td class="tg-0lax">18.05</td>
  </tr>
  <tr>
    <td class="tg-0lax">584</td>
    <td class="tg-0lax">15.52</td>
    <td class="tg-0lax">10.33</td>
    <td class="tg-0lax">24.83</td>
    <td class="tg-0lax">16.71</td>
  </tr>
  <tr>
    <td class="tg-0lax">596</td>
    <td class="tg-0lax">12.84</td>
    <td class="tg-0lax">8.45</td>
    <td class="tg-0lax">20.46</td>
    <td class="tg-0lax">13.85</td>
  </tr>
  <tr>
    <td class="tg-0lax">mean</td>
    <td class="tg-0lax">14.35</td>
    <td class="tg-0lax">9.41</td>
    <td class="tg-0lax">23.87</td>
    <td class="tg-0lax">16.18</td>
  </tr>
</tbody>
</table>

# Discussion

I don't believe the results are accurate. 
I could not find a single paper with better results than my described approach, but because I only spent a relatively short amount of time on this task, it is hard to believe to achieve the best results with simple models. I believe this is because I removed all data where missing cbg values are contained.
I checked out the [repository](https://gitlab.eecs.umich.edu/mld3/deep-residual-time-series-forecasting) to compare where my mistakes are but couldn't find any.


# Pytorch Forecasting

I also created a notebook, using the library to create a temporal fusion transformer. 
I did not focus on the results of this approach because I just adapted the tutorial on the docs, which would not have shown my skills in Deep Learning. 
The resulted model even outperformed all previously mentioned models. 
If I would have had more time to solve this task, I would have studied the methods of this library further to improve my results.


# Future Work

Due to the limited time I had to solve this task, there is much more to try out. 
I only implemented simple methods and ideas. 
Here is a list of things I would do to improve the given results. 

1. Data Processing:
   * Using the ARIMA model to interpolate the missing feature values instead of using splines
   * Since ARIMA is pretty powerful, I believe that it can supplement more accurate values
   * Data Augmentation: I don't know any method how it is possible, but I could spend some research
2. Hyperparameter Tuning:
   * I only "guessed" good parameters instead of using well-known techniques like grid search or bayesian search
   * There is the library [optuna](https://optuna.org/) which offers such functionality
   * Alternatively, I could just use sklearn
3. Regularization: 
   * As I mentioned in the Data preparation section, there exist a few outliers
   * To not overfit them, one can use regularization methods (dropout, batch norm, layer norm, etc.)
4. Temporal Fusion Transformer
   * As mentioned, the transformer can outperform other models
5. N-fold cross validation:
   * I did not include this, because I had limited computational resources