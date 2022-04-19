# API Introduction of `CurveFigureAnalysis`

## Above all

## Index

[TOC]

# Objects

## Line Classes

### `LineFigure`

`LineFigrue` is the base class, aiming at extracting figure from the pic.

#### Class Method

##### `__init__`

```python
# constructors
@abc.abstractmethod
def __init__(self, rawPic, givenPic=None, picLabel=None)
```

Actually we attempt to initiate the object with targeted picture. 

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `fromFile`

This method would read in the pic from referenced file properly structured.

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Calculating Functions

##### `IsBinPic_Valid`

This function would judge the picture is usable or not.

A picture is usable or not depends on the valid points in the pic, or rather the white point in a binary pic.

And this function would be used to certificate the size of output set.

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `GetColorInterval`

This function using interval method to calculate the color of the line in the pic.

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `ColorHistCalc`

Display function, calculating the color hist of the picture through 4 cannels.

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `IsBinPicNormalized`

This function would judge the pic is normalized or not.

Since we'd get pic with lighter or darker background, sometimes we get binary pic with white background.

So turn the white background into black is called normalize. 

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Binary Pic Output Functions

These functions are the core functions of `LineFigure`, and each function would return a targeted binary picture using different method.

##### `BinPic_TotalFilter`

This function would use the interval method to filter the channels and return a binary pic for each channel.

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `BinPic_getCannyPic`

Wrapper of `opencv` `imgCanny` function.

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `BinPic_AdaptiveThresh`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `BinPic_imgOverlay`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `BinPic_SmoothOutput`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `BinPic_CentralDivide`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `BinPic_HEDMethod_Processed`

This function would use the `hed` method, to transform the raw pic into binary pic.

Please read 

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `BinPic_SetGetter`

This functions is the final function of `BinPic_`, it would return the set containing all valid pics.

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |


#### Util Functions

##### `getMask`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### Sub Classes

#### `BrokenLineFigure`

#### `CurveFigure`

## Info Extractor

### `PointDetector`

#### Class Method

##### `__init__`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `FromBinPic`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Correction Functions

##### `AlterInit_Correction`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `PointsTrans`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `PointsTrans_Targeted`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `MissedPoints_Fill_Interp`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Correction with Peaks

##### `Peak_Correction`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `Trough_Correction`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `PeakAndTrough_Correction_All`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Get Results

##### by percent

###### `GetResult_Specific_ByPercentage`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

###### `GetResult_TarVector_ByPercentage`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### by x

###### `GetResult_Specific_ByX_Centralized`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

###### `GetResult_Specific_ByX_Centralized_Fitted_Insert`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

###### `GetResult_Specific_ByX_Centralized_Interp_Insert`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

###### `GetResult_Specific_ByX_Clump`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

###### `GetResult_TarVector_ByX`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Slice Method

##### `GetSlice_ByX`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `Display_Sliced`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `Display_All`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `FigureInfo`

#### Class Method

##### `__init__`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Get Current `PointDetector`

##### `Get_Poi_Hierarchy`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Output Functions

##### `Output_Central_Interp_Hierarchy`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `Output_Mean_Hierarchy`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `Output_Central_Fit_Correction`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `Output_Multi_func`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `Output_Bad_Result`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `ResultProducer`

#### Class Method

##### `__init__`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `__getitem__`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `__len__`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Set Functions

##### `SetTestSet`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

#### Produce Functions

##### `ProduceToSheet`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

##### `ProduceToExcel`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

## Figure Box

### Util Functions

### `biggestContour`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `toPoints`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `reorder`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### Pic Transfer

#### `TransPicGetter`

#### `PicWrapper`

### Figure Box

#### `getMaxBox`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

## Exception Classes

### `OutputErrorOfBadQuality`

### `OutputErrorOfBlank`

### `OutputErrorOfSpecificPos`

# Function

## Point Detect Functions

### `LinePointDetectCentralize`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `LinePointDetectCentralizeAmplified`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `DetectPointHarrisMethod`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `DetectPointGoodFeatureMethod`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

## Display Functions

### `AdaptiveShow`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `DispPoints`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `Display_AllPoints`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `Display_AllPoints_Test`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

## HED Functions and Wrapper


### `PicTrans2HEDInput`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `HEDDetect`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |

### `HEDDetectSave`

| param | type | description |
| ----- | ---- | ----------- |
|       |      |             |
|       |      |             |
|       |      |             |