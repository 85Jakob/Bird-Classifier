# 525 Bird Species Classifier

## Project Summary 

To aid with wildlfie conservation I aim to create a robust machine learning models that can correctly identify a bird's species. The dataset being used for this project is the [100-bird-species dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species/code?datasetId=534640&sortBy=voteCount) which can be found on kaggle.com. The current dataset contains images for 524 bird species. Each image contains a bird that is perfectly centered. To better replicate photos taken by wildlife cameras, the images will be cropped or zommed, so that some of the images of the birds are not as clear. This report will test to see how well [ResNet50](https://keras.io/api/applications/resnet/) can preform at this task.  

## Problem Statement 

The main objective is to create a robust model that has high accuracy at correctly classifying bird images and handling a wide variation of image qualities.   
 
To train the models I will be using a dataset of 89,885 images of 525 bird species from Kaggle. To create a more varied and balanced dataset an additional 53,440 augmented images will be added, bringing the total set to 143,325 images. The augmented images are images from the base set but with a transformation applied to them.  

 ResNet50 will be trained on a 64% of this dataset with 16% set aside for validation and 20% for testing.  

 ## Dataset 

The dataset being used contains 89,885 colored images of birds. While most of the images in the dataset are 224x224 pixels, the set does contain 201 images of various sizes. This means that the dataset being used contains approximatly 89,885 incidences and 50,176 features. The dataset is split up into three directories, one for training, testing, and validation. Each of these directories contains 525 different folders for each bird species. One of the 525 folders labeled "loony birds" does not contain images of bird but instead has pictures of polaticians. The species folders in each directory are the same. However the images for each species in each directory are different. During the exploratory data analysis it was determined that the data set was imbalanced. To balance out the data set an additional 53,440 images were added.  

![Screenshot from 2023-12-21 11-08-07](https://github.com/85Jakob/Bird-Classifier/assets/24831044/ca3b6bba-7c9d-475b-93b2-6917976c3c17)  

### Training Directory  
The training directory has 84,635 images split up into 525 folders for each bird species. Each speceies folder contained between 130 to 263 images. 

### Test and Validation Directory  
The test and validation directory are setup in the exact same way each containing 2,625 images split up into 525 folders for each spieces. The sub folders all contained 5 images. 

## Exploratory Data Analysis   
### Data Split  
The dataset is already split into three sets; train, test, and valid. The training set has 84,635 images, the test set has 2,625, and the validation set has 2,625. This means that 92.4% of our data is in the training set. This is not an ideal split. To make a more fair split to reduce risks of overfitting and to have a more reliable validation and testing scores we will need to change the split to be closer to 64% training, 16% validation, and 20% testing.    

![Screenshot from 2023-12-21 11-16-11](https://github.com/85Jakob/Bird-Classifier/assets/24831044/6bb04d4d-36bf-472e-8c8f-62b04d9ddf43)  

Each set contains 525 species or classes. Each set also contains the same classes, however the file name for Parakett Auklet in the test and training set contained a typing error and had an extra space in it. The space was removed in order to match the name listed in the validation set

### Image Size  
The images in this dataset are not all the same size. While the majority of the images are 224x224 there are 211 images that are either smaller or larger in shape. We will resize all the images to a standard size of either 128x128 to have a consistant input data and to minimize the use of memory while training the model. 

Below shows the distribution of the height, width, and number of color channels for the training set.  
![Screenshot from 2023-12-21 11-20-14](https://github.com/85Jakob/Bird-Classifier/assets/24831044/1914755c-fae1-4e87-95fc-a0c8487323a5)  

The following charts show the distribution of the height, width, and number of color channels for the validation set.  
![Screenshot from 2023-12-21 11-20-47](https://github.com/85Jakob/Bird-Classifier/assets/24831044/eb3e79f7-5dd9-44bb-aafd-0ff2db5af7fb)  

The next charts show the distribution of the height, width, and number of color channels for the testing set.  
![Screenshot from 2023-12-21 11-21-08](https://github.com/85Jakob/Bird-Classifier/assets/24831044/a7e3b418-83c3-4249-a983-cc8bdbf6d005)  

### Sample Size of Each Class  
 
The training set is imbalanced as seen in the histogram below. The sample size for each class ranges from 130 images to 263 images. To avoid biases in or model towards classes with larger sample sizes we will add more images to the classes with smaller samples sizes. Every class in the training and validation set have exactly 5 sample images.     

Below is the distribution of the sample size for each class in the training set.  
![Screenshot from 2023-11-17 09-44-21](https://github.com/85Jakob/Bird-Classifier/assets/24831044/41e14f74-6220-47be-96fa-911adbb36c63)  

Species with the most images:    
- Rufous Trepe - 263 images  
- House Finch - 248 images  
- Ovenbird - 233 images  
 
Species with the least amount of images:  
- Eurasian Golden Oriole - 130 images  
- Red  Trail Thrush - 130 images  
- Snowy Plover - 130 images  

Below shows that each class in the validation and testing set contain 5 images.  
![Screenshot from 2023-12-21 11-35-29](https://github.com/85Jakob/Bird-Classifier/assets/24831044/bef85f0b-8b88-4788-9495-90fad1cc7011)  
![Screenshot from 2023-12-21 11-35-53](https://github.com/85Jakob/Bird-Classifier/assets/24831044/d7a8dcb7-73ad-4ba3-842e-65fe796eef71)  

### Looking at the Data
Cycling through samples of the data shows that the birds in each image take up at least 50% of the image space. We also can see that all the images are in color. In preprocessing when we add aditional images we will shift, rotate and apply a zoom so that the birds are not perfectly center. This will hopefully create samples that resemble more photos taken out in nature.

![Screenshot from 2023-11-17 10-00-18](https://github.com/85Jakob/Bird-Classifier/assets/24831044/9487e287-b659-466a-b409-89fd0b411cab)  

### Balancing Dataset
To balance the dataset we will combine the test, training, and validation set. Next any class with less than 273 samples will have augmented images added to it untill it contains exactly 273 images. 

### Image Augmentation  
The image augmentations used to fill up the folders include a random zoom, flip, rotation or a combo of the three. Below shows an example of the different effects added.

Original Photo   
![048](https://github.com/85Jakob/Bird-Classifier/assets/24831044/4dda998f-8339-4ca6-9d89-95e0033fc400)  

flip  
![SUPERB STARLING_original_048 jpg_94484b03-bfc3-420b-935d-3d43d937b9db](https://github.com/85Jakob/Bird-Classifier/assets/24831044/17d5765b-983a-4e58-9263-30339d2ca9e9)  

Zoom   
![SUPERB STARLING_original_048 jpg_290fb15f-2d51-464d-b943-c6ce9ba8fdee](https://github.com/85Jakob/Bird-Classifier/assets/24831044/ae15dabe-caa8-40e8-993d-319a63090fdf)  

### Creating a New Split
Now that all the images are in the same directory we can use train_test_split() to establish our own split percentages.  
![Screenshot from 2023-12-21 11-52-09](https://github.com/85Jakob/Bird-Classifier/assets/24831044/e647c830-8330-496d-b290-1380e9683edf)  

### Resizing, Shuffling, Batching, and Normaliztion

All the images were resized to be 128x128. This is to allows for a consist input into our neural network. The images were batched into 32 seprate sets. Batching is done to reduce the amount of image data loaded into memory. Batching will also increase the speed of training as it will only need to store the error values for the images in the batch. After the batch is completed the model will update its weights. The images were also shuffled, as this helps prevent the model from becoming overfited to the data. The shuffling allows our batches to be a better representation of our dataset. Lastly the data is normalized to become a value between 0 and 1. Normalization is important because by reducing the range of the imput values we can minimize the results in our gradient calculations.  

### ResNet50 Model  
The parameters for our model are input_shape=(128, 128, 3), weights='imagenet', and pooling='avg'.  
After going through the ResNet base a Dense, Normalization and Dropout layer were added. These layers will prevent the model from overfitting to the training data and prevent any gradients from getting too large. 

# Results
The model reached a Training accuracy of 99% with a Validation accuracy of 88%  
![image](https://github.com/85Jakob/Bird-Classifier/assets/24831044/2fbada1d-c8c6-4c66-9936-c5321aa1ab3e)  

The model also had a Training loss of 0.015 and Validation loss of 0.51  
![image](https://github.com/85Jakob/Bird-Classifier/assets/24831044/6db5f8dc-5afd-4b99-9856-cf230f41afdb)  

When the model was tested on the Test set it achieved an accuracy of 89%. The following are some predictions the model made on the test set.   

![Screenshot from 2023-12-21 12-12-18](https://github.com/85Jakob/Bird-Classifier/assets/24831044/ede1f284-bae0-4d15-815a-e70e4aceba9e)    

The following are the scores per class  

```
                               precision    recall  f1-score   support

              ABBOTTS BABBLER       0.49      0.74      0.59        35
                ABBOTTS BOOBY       0.67      0.82      0.74        51
   ABYSSINIAN GROUND HORNBILL       0.92      0.82      0.87        56
        AFRICAN CROWNED CRANE       0.98      0.97      0.97        58
       AFRICAN EMERALD CUCKOO       0.75      0.75      0.75        53
            AFRICAN FIREFINCH       0.88      0.84      0.86        51
       AFRICAN OYSTER CATCHER       0.96      0.92      0.94        59
        AFRICAN PIED HORNBILL       0.83      0.72      0.77        60
          AFRICAN PYGMY GOOSE       0.98      0.94      0.96        62
                    ALBATROSS       0.81      0.90      0.85        48
               ALBERTS TOWHEE       0.84      0.94      0.88        49
         ALEXANDRINE PARAKEET       0.89      0.94      0.92        52
                ALPINE CHOUGH       0.88      0.89      0.89        66
        ALTAMIRA YELLOWTHROAT       0.85      0.89      0.87        53
              AMERICAN AVOCET       0.97      0.97      0.97        60
             AMERICAN BITTERN       0.89      0.91      0.90        56
                AMERICAN COOT       0.94      0.89      0.92        56
              AMERICAN DIPPER       0.92      0.85      0.88        54
            AMERICAN FLAMINGO       0.98      0.96      0.97        49
           AMERICAN GOLDFINCH       0.90      0.86      0.88        50
             AMERICAN KESTREL       0.97      0.94      0.95        62
               AMERICAN PIPIT       0.84      0.87      0.85        47
            AMERICAN REDSTART       0.91      0.91      0.91        54
               AMERICAN ROBIN       0.96      0.89      0.92        55
              AMERICAN WIGEON       0.93      0.82      0.87        50
            AMETHYST WOODSTAR       0.81      0.81      0.81        57
                 ANDEAN GOOSE       0.86      0.95      0.90        44
               ANDEAN LAPWING       0.93      0.76      0.83        66
                ANDEAN SISKIN       0.68      0.74      0.71        54
                      ANHINGA       0.84      0.89      0.86        57
                     ANIANIAU       0.86      0.79      0.82        56
            ANNAS HUMMINGBIRD       0.82      0.80      0.81        59
                      ANTBIRD       0.84      0.76      0.80        62
           ANTILLEAN EUPHONIA       0.89      0.83      0.86        58
                      APAPANE       0.96      0.84      0.90        56
                  APOSTLEBIRD       0.84      0.88      0.86        56
              ARARIPE MANAKIN       0.96      0.98      0.97        55
            ASHY STORM PETREL       0.69      0.83      0.75        42
              ASHY THRUSHBIRD       0.86      0.74      0.79        65
           ASIAN CRESTED IBIS       0.96      0.95      0.95        56
           ASIAN DOLLARD BIRD       0.90      0.92      0.91        62
        ASIAN GREEN BEE EATER       0.95      0.97      0.96        61
         ASIAN OPENBILL STORK       0.82      0.85      0.84        48
                AUCKLAND SHAQ       0.82      0.62      0.71        68
            AUSTRAL CANASTERO       0.72      0.69      0.71        61
         AUSTRALASIAN FIGBIRD       0.91      0.84      0.87        61
                     AVADAVAT       0.93      0.88      0.90        58
             AZARAS SPINETAIL       0.87      0.82      0.84        56
         AZURE BREASTED PITTA       0.95      0.96      0.96        56
                    AZURE JAY       0.85      0.80      0.82        50
                AZURE TANAGER       0.90      0.92      0.91        51
                    AZURE TIT       0.94      0.94      0.94        63
                  BAIKAL TEAL       0.88      0.92      0.90        53
                   BALD EAGLE       0.89      0.86      0.88        59
                    BALD IBIS       0.88      0.86      0.87        44
                BALI STARLING       0.94      0.96      0.95        47
             BALTIMORE ORIOLE       0.81      0.87      0.84        68
                   BANANAQUIT       0.87      0.80      0.83        56
             BAND TAILED GUAN       0.73      0.80      0.76        44
             BANDED BROADBILL       0.95      0.91      0.93        66
                  BANDED PITA       0.99      1.00      0.99        73
                 BANDED STILT       0.78      0.88      0.83        43
            BAR-TAILED GODWIT       0.89      0.89      0.89        55
                     BARN OWL       0.94      0.96      0.95        50
                 BARN SWALLOW       0.88      0.81      0.84        52
              BARRED PUFFBIRD       0.85      0.85      0.85        48
            BARROWS GOLDENEYE       0.97      0.95      0.96        60
         BAY-BREASTED WARBLER       0.89      0.96      0.92        51
               BEARDED BARBET       0.98      0.94      0.96        62
             BEARDED BELLBIRD       0.95      0.95      0.95        66
             BEARDED REEDLING       0.91      0.87      0.89        47
            BELTED KINGFISHER       0.89      0.98      0.93        51
             BIRD OF PARADISE       0.96      0.91      0.93        55
   BLACK AND YELLOW BROADBILL       0.91      0.98      0.94        51
                   BLACK BAZA       0.86      0.83      0.84        59
      BLACK BREASTED PUFFBIRD       0.91      0.95      0.93        55
                BLACK COCKATO       0.87      0.88      0.88        52
        BLACK FACED SPOONBILL       0.91      0.90      0.90        58
              BLACK FRANCOLIN       0.94      0.86      0.90        58
          BLACK HEADED CAIQUE       0.96      0.94      0.95        49
           BLACK NECKED STILT       0.87      0.84      0.85        56
                BLACK SKIMMER       0.89      0.84      0.86        49
                   BLACK SWAN       0.96      0.98      0.97        48
             BLACK TAIL CRAKE       0.96      1.00      0.98        55
       BLACK THROATED BUSHTIT       0.93      0.90      0.92        63
          BLACK THROATED HUET       0.89      0.83      0.86        59
       BLACK THROATED WARBLER       0.72      0.84      0.77        49
      BLACK VENTED SHEARWATER       0.78      0.80      0.79        49
                BLACK VULTURE       0.81      0.79      0.80        43
       BLACK-CAPPED CHICKADEE       0.82      0.98      0.90        48
           BLACK-NECKED GREBE       0.94      0.96      0.95        49
       BLACK-THROATED SPARROW       0.95      0.85      0.90        62
         BLACKBURNIAM WARBLER       0.82      0.82      0.82        51
    BLONDE CRESTED WOODPECKER       0.93      0.95      0.94        58
               BLOOD PHEASANT       0.98      0.98      0.98        58
                    BLUE COAU       0.80      0.87      0.83        55
                  BLUE DACNIS       0.81      0.88      0.84        58
        BLUE GRAY GNATCATCHER       0.84      0.77      0.80        47
                BLUE GROSBEAK       0.97      0.90      0.93        62
                  BLUE GROUSE       0.70      0.86      0.77        49
                   BLUE HERON       0.86      0.88      0.87        65
                 BLUE MALKOHA       0.81      0.76      0.78        74
    BLUE THROATED PIPING GUAN       0.68      0.74      0.71        57
       BLUE THROATED TOUCANET       0.87      0.87      0.87        61
                     BOBOLINK       0.73      0.78      0.76        46
          BORNEAN BRISTLEHEAD       0.98      0.90      0.94        49
             BORNEAN LEAFBIRD       0.90      0.85      0.88        55
             BORNEAN PHEASANT       0.92      0.89      0.91        54
             BRANDT CORMARANT       0.85      0.82      0.84        68
            BREWERS BLACKBIRD       0.68      0.57      0.62        47
                BROWN CREPPER       0.92      0.86      0.89        57
         BROWN HEADED COWBIRD       0.65      0.58      0.62        48
                  BROWN NOODY       0.96      0.88      0.92        57
               BROWN THRASHER       0.88      0.95      0.92        64
                   BUFFLEHEAD       0.82      0.81      0.82        52
             BULWERS PHEASANT       0.95      0.95      0.95        41
            BURCHELLS COURSER       0.93      0.86      0.89        49
                  BUSH TURKEY       0.89      0.91      0.90        54
           CAATINGA CACHOLOTE       0.89      0.93      0.91        42
              CABOTS TRAGOPAN       0.87      0.84      0.86        57
                  CACTUS WREN       0.85      0.84      0.85        56
            CALIFORNIA CONDOR       0.80      0.80      0.80        54
              CALIFORNIA GULL       0.80      0.80      0.80        40
             CALIFORNIA QUAIL       0.92      0.94      0.93        63
                CAMPO FLICKER       0.94      0.98      0.96        52
                       CANARY       0.77      0.82      0.79        61
                   CANVASBACK       0.82      0.88      0.85        52
         CAPE GLOSSY STARLING       0.93      0.95      0.94        55
                CAPE LONGCLAW       0.91      0.83      0.87        48
             CAPE MAY WARBLER       0.75      0.82      0.78        50
             CAPE ROCK THRUSH       0.87      0.93      0.90        56
                 CAPPED HERON       0.96      0.92      0.94        49
                 CAPUCHINBIRD       0.88      0.95      0.91        55
            CARMINE BEE-EATER       0.92      0.94      0.93        50
                 CASPIAN TERN       0.89      0.92      0.90        51
                    CASSOWARY       0.95      0.90      0.92        61
                CEDAR WAXWING       0.91      0.93      0.92        55
             CERULEAN WARBLER       0.93      0.89      0.91        63
              CHARA DE COLLAR       0.85      0.82      0.84        56
              CHATTERING LORY       0.89      1.00      0.94        51
    CHESTNET BELLIED EUPHONIA       0.94      0.95      0.94        61
       CHESTNUT WINGED CUCKOO       0.78      0.90      0.83        48
     CHINESE BAMBOO PARTRIDGE       0.91      0.95      0.93        44
           CHINESE POND HERON       0.89      0.83      0.86        48
             CHIPPING SPARROW       0.85      0.85      0.85        60
              CHUCAO TAPACULO       0.91      0.88      0.89        67
             CHUKAR PARTRIDGE       0.96      0.93      0.95        56
              CINNAMON ATTILA       0.82      0.88      0.84        56
          CINNAMON FLYCATCHER       0.87      0.92      0.89        60
                CINNAMON TEAL       0.86      0.96      0.91        52
                 CLARKS GREBE       0.95      0.93      0.94        56
            CLARKS NUTCRACKER       0.89      0.87      0.88        55
            COCK OF THE  ROCK       0.94      0.95      0.94        61
                     COCKATOO       0.98      1.00      0.99        57
             COLLARED ARACARI       0.87      0.90      0.88        58
       COLLARED CRESCENTCHEST       0.74      0.84      0.79        50
             COMMON FIRECREST       0.94      0.96      0.95        51
               COMMON GRACKLE       0.77      0.77      0.77        48
          COMMON HOUSE MARTIN       0.73      0.84      0.78        55
                  COMMON IORA       0.74      0.96      0.83        50
                  COMMON LOON       0.96      0.91      0.93        54
              COMMON POORWILL       0.92      0.87      0.90        54
              COMMON STARLING       0.94      0.90      0.92        50
           COPPERSMITH BARBET       0.84      0.98      0.90        57
        COPPERY TAILED COUCAL       0.85      0.75      0.80        60
                  CRAB PLOVER       0.96      0.93      0.95        46
                   CRANE HAWK       0.80      0.80      0.80        59
     CREAM COLORED WOODPECKER       0.88      0.90      0.89        59
               CRESTED AUKLET       0.92      0.97      0.94        61
             CRESTED CARACARA       0.89      0.88      0.89        67
                 CRESTED COUA       0.86      0.81      0.84        63
             CRESTED FIREBACK       0.89      0.89      0.89        65
           CRESTED KINGFISHER       0.94      0.94      0.94        53
             CRESTED NUTHATCH       0.93      0.78      0.85        54
           CRESTED OROPENDOLA       0.86      0.69      0.77        55
        CRESTED SERPENT EAGLE       0.80      0.62      0.70        58
            CRESTED SHRIKETIT       0.88      0.89      0.89        66
       CRESTED WOOD PARTRIDGE       0.92      0.95      0.94        61
                 CRIMSON CHAT       0.90      0.92      0.91        61
              CRIMSON SUNBIRD       0.93      0.96      0.94        53
                         CROW       0.77      0.82      0.79        66
                   CUBAN TODY       0.98      1.00      0.99        56
                 CUBAN TROGON       0.90      0.87      0.88        53
         CURL CRESTED ARACURI       0.93      0.90      0.92        62
             D-ARNAUDS BARBET       0.88      0.85      0.87        54
            DALMATIAN PELICAN       0.98      0.90      0.94        58
        DARJEELING WOODPECKER       0.84      0.91      0.87        46
              DARK EYED JUNCO       0.95      0.85      0.90        66
             DAURIAN REDSTART       0.92      0.87      0.90        69
             DEMOISELLE CRANE       0.93      0.80      0.86        51
          DOUBLE BARRED FINCH       1.00      0.94      0.97        49
     DOUBLE BRESTED CORMARANT       0.77      0.76      0.76        49
       DOUBLE EYED FIG PARROT       0.86      0.90      0.88        49
             DOWNY WOODPECKER       0.89      0.93      0.91        60
                       DUNLIN       0.93      0.94      0.93        53
                   DUSKY LORY       0.98      0.93      0.95        45
                  DUSKY ROBIN       0.66      0.79      0.72        47
                   EARED PITA       0.98      0.83      0.90        66
             EASTERN BLUEBIRD       0.94      0.95      0.95        64
           EASTERN BLUEBONNET       0.95      0.97      0.96        61
        EASTERN GOLDEN WEAVER       0.81      0.70      0.75        50
           EASTERN MEADOWLARK       0.94      0.94      0.94        53
              EASTERN ROSELLA       0.94      0.89      0.92        56
                EASTERN TOWEE       0.89      0.94      0.92        52
        EASTERN WIP POOR WILL       0.83      0.96      0.89        45
         EASTERN YELLOW ROBIN       0.86      0.92      0.89        53
          ECUADORIAN HILLSTAR       1.00      0.85      0.92        53
               EGYPTIAN GOOSE       0.80      0.88      0.84        42
               ELEGANT TROGON       0.81      0.84      0.83        57
            ELLIOTS  PHEASANT       0.92      0.93      0.93        61
              EMERALD TANAGER       0.93      0.95      0.94        59
              EMPEROR PENGUIN       0.98      0.97      0.97        58
                          EMU       0.94      0.90      0.92        52
                 ENGGANO MYNA       0.85      0.85      0.85        54
           EURASIAN BULLFINCH       0.98      0.95      0.97        66
       EURASIAN GOLDEN ORIOLE       0.87      0.82      0.84        56
              EURASIAN MAGPIE       0.98      0.81      0.88        62
           EUROPEAN GOLDFINCH       0.98      0.96      0.97        51
         EUROPEAN TURTLE DOVE       0.94      0.92      0.93        48
             EVENING GROSBEAK       0.91      0.96      0.93        50
               FAIRY BLUEBIRD       0.85      0.85      0.85        61
                FAIRY PENGUIN       0.89      0.91      0.90        45
                   FAIRY TERN       0.91      0.85      0.88        60
             FAN TAILED WIDOW       0.78      0.91      0.84        54
               FASCIATED WREN       0.64      0.71      0.67        48
                FIERY MINIVET       0.88      0.83      0.85        63
            FIORDLAND PENGUIN       0.82      0.91      0.86        55
        FIRE TAILLED MYZORNIS       0.93      0.96      0.94        53
              FLAME BOWERBIRD       1.00      0.90      0.95        51
                FLAME TANAGER       0.84      0.87      0.86        55
               FOREST WAGTAIL       0.84      0.84      0.84        61
                      FRIGATE       0.94      0.92      0.93        63
            FRILL BACK PIGEON       0.86      0.94      0.89        63
                GAMBELS QUAIL       0.92      0.94      0.93        62
           GANG GANG COCKATOO       0.91      1.00      0.95        53
              GILA WOODPECKER       0.91      0.89      0.90        46
               GILDED FLICKER       0.88      0.82      0.85        56
                  GLOSSY IBIS       0.94      0.90      0.92        50
                 GO AWAY BIRD       0.83      0.91      0.87        57
            GOLD WING WARBLER       0.91      0.94      0.92        51
            GOLDEN BOWER BIRD       0.88      0.98      0.93        59
       GOLDEN CHEEKED WARBLER       0.94      0.84      0.89        58
          GOLDEN CHLOROPHONIA       0.93      0.89      0.91        46
                 GOLDEN EAGLE       0.78      0.87      0.82        54
              GOLDEN PARAKEET       0.94      0.98      0.96        52
              GOLDEN PHEASANT       0.98      0.97      0.97        60
                 GOLDEN PIPIT       0.94      0.94      0.94        53
               GOULDIAN FINCH       0.95      0.95      0.95        42
                     GRANDALA       0.96      0.92      0.94        51
                 GRAY CATBIRD       0.78      0.90      0.83        50
                GRAY KINGBIRD       0.72      0.71      0.72        48
               GRAY PARTRIDGE       0.88      0.91      0.90        67
                  GREAT ARGUS       0.80      0.92      0.86        36
               GREAT GRAY OWL       0.91      0.91      0.91        54
                GREAT JACAMAR       0.88      0.98      0.93        47
               GREAT KISKADEE       0.90      0.85      0.88        55
                  GREAT POTOO       0.91      0.91      0.91        65
                GREAT TINAMOU       0.92      0.85      0.89        55
                 GREAT XENOPS       0.69      0.70      0.70        44
                GREATER PEWEE       0.71      0.83      0.77        60
      GREATER PRAIRIE CHICKEN       1.00      0.91      0.95        44
          GREATOR SAGE GROUSE       0.93      0.87      0.90        60
              GREEN BROADBILL       0.96      0.88      0.92        58
                    GREEN JAY       0.95      0.90      0.92        59
                 GREEN MAGPIE       0.97      0.90      0.93        62
            GREEN WINGED DOVE       0.89      0.91      0.90        54
            GREY CUCKOOSHRIKE       0.69      0.79      0.74        53
       GREY HEADED CHACHALACA       0.83      0.83      0.83        48
       GREY HEADED FISH EAGLE       0.70      0.79      0.75        48
                  GREY PLOVER       0.98      0.89      0.94        57
            GROVED BILLED ANI       0.77      0.82      0.79        60
                GUINEA TURACO       0.94      0.75      0.83        64
                   GUINEAFOWL       0.94      0.83      0.88        53
                GURNEYS PITTA       1.00      0.98      0.99        56
                    GYRFALCON       0.82      0.82      0.82        51
                     HAMERKOP       0.92      0.92      0.92        52
               HARLEQUIN DUCK       0.98      1.00      0.99        58
              HARLEQUIN QUAIL       0.89      0.95      0.92        44
                  HARPY EAGLE       0.84      0.89      0.86        54
               HAWAIIAN GOOSE       0.98      0.98      0.98        53
                     HAWFINCH       0.98      0.89      0.93        55
                 HELMET VANGA       0.93      0.93      0.93        56
              HEPATIC TANAGER       0.87      0.80      0.83        56
           HIMALAYAN BLUETAIL       0.88      0.95      0.91        56
              HIMALAYAN MONAL       0.93      1.00      0.96        39
                      HOATZIN       0.88      0.89      0.88        56
             HOODED MERGANSER       0.87      0.94      0.90        50
                      HOOPOES       0.96      0.93      0.95        58
                  HORNED GUAN       0.85      0.91      0.88        64
                  HORNED LARK       0.98      0.95      0.96        56
                HORNED SUNGEM       0.97      0.94      0.95        64
                  HOUSE FINCH       0.84      0.89      0.86        57
                HOUSE SPARROW       0.75      0.84      0.79        50
               HYACINTH MACAW       0.98      0.96      0.97        57
               IBERIAN MAGPIE       0.85      0.98      0.91        54
                     IBISBILL       0.94      0.88      0.91        52
                IMPERIAL SHAQ       0.70      0.56      0.62        62
                    INCA TERN       0.98      0.94      0.96        49
               INDIAN BUSTARD       0.98      0.92      0.95        53
                 INDIAN PITTA       0.95      0.95      0.95        58
                INDIAN ROLLER       0.93      0.88      0.90        48
               INDIAN VULTURE       0.72      0.78      0.75        65
               INDIGO BUNTING       0.79      0.88      0.83        56
            INDIGO FLYCATCHER       0.91      0.98      0.94        59
              INLAND DOTTEREL       0.92      0.85      0.88        52
         IVORY BILLED ARACARI       0.82      0.95      0.88        43
                   IVORY GULL       0.79      0.84      0.81        55
                          IWI       0.83      0.90      0.86        48
                       JABIRU       0.96      0.90      0.93        50
                   JACK SNIPE       0.93      0.97      0.95        59
               JACOBIN PIGEON       0.93      0.95      0.94        59
             JANDAYA PARAKEET       0.91      0.91      0.91        46
               JAPANESE ROBIN       0.89      0.94      0.92        54
                 JAVA SPARROW       0.94      0.96      0.95        50
            JOCOTOCO ANTPITTA       0.94      0.98      0.96        45
                         KAGU       0.91      0.91      0.91        69
                       KAKAPO       1.00      0.94      0.97        53
                     KILLDEAR       0.94      0.94      0.94        53
                   KING EIDER       0.98      0.90      0.94        49
                 KING VULTURE       0.93      0.95      0.94        58
                         KIWI       0.95      1.00      0.97        58
             KNOB BILLED DUCK       0.89      0.78      0.83        51
                   KOOKABURRA       0.85      0.88      0.87        52
                 LARK BUNTING       0.81      0.81      0.81        48
                LAUGHING GULL       0.94      0.93      0.93        54
               LAZULI BUNTING       0.91      0.98      0.94        49
              LESSER ADJUTANT       0.82      0.92      0.87        50
                 LILAC ROLLER       0.93      0.98      0.96        58
                      LIMPKIN       0.95      0.95      0.95        59
                   LITTLE AUK       0.84      0.88      0.86        67
            LOGGERHEAD SHRIKE       0.88      0.81      0.84        53
               LONG-EARED OWL       0.87      0.90      0.88        51
                 LOONEY BIRDS       1.00      1.00      1.00        54
          LUCIFER HUMMINGBIRD       0.92      0.89      0.90        62
                 MAGPIE GOOSE       0.86      0.89      0.88        57
             MALABAR HORNBILL       0.78      0.81      0.80        53
         MALACHITE KINGFISHER       0.91      0.93      0.92        56
           MALAGASY WHITE EYE       0.85      0.74      0.79        46
                        MALEO       0.98      0.93      0.96        70
                 MALLARD DUCK       0.98      0.97      0.98        62
                 MANDRIN DUCK       1.00      0.98      0.99        60
              MANGROVE CUCKOO       0.76      0.89      0.82        54
                MARABOU STORK       0.89      0.85      0.87        48
              MASKED BOBWHITE       0.86      0.88      0.87        68
                 MASKED BOOBY       0.81      0.85      0.83        60
               MASKED LAPWING       0.76      0.95      0.85        44
               MCKAYS BUNTING       0.96      0.90      0.93        50
                       MERLIN       0.77      0.84      0.80        43
             MIKADO  PHEASANT       0.93      0.85      0.89        47
               MILITARY MACAW       0.98      0.93      0.96        59
                MOURNING DOVE       0.85      0.94      0.89        49
                         MYNA       0.84      0.92      0.88        51
               NICOBAR PIGEON       0.96      0.98      0.97        52
              NOISY FRIARBIRD       0.73      0.73      0.73        51
NORTHERN BEARDLESS TYRANNULET       0.78      0.69      0.73        61
            NORTHERN CARDINAL       0.88      0.86      0.87        51
             NORTHERN FLICKER       0.80      0.82      0.81        55
              NORTHERN FULMAR       0.90      0.74      0.81        61
              NORTHERN GANNET       0.89      0.80      0.84        59
             NORTHERN GOSHAWK       0.61      0.74      0.67        53
              NORTHERN JACANA       0.96      0.93      0.95        57
         NORTHERN MOCKINGBIRD       0.71      0.88      0.79        51
              NORTHERN PARULA       0.90      0.77      0.83        56
          NORTHERN RED BISHOP       0.93      0.96      0.94        53
            NORTHERN SHOVELER       0.94      0.90      0.92        52
             OCELLATED TURKEY       0.96      0.99      0.97        70
                      OILBIRD       0.93      0.98      0.95        43
                 OKINAWA RAIL       1.00      0.91      0.95        68
       ORANGE BREASTED TROGON       0.93      0.87      0.90        47
       ORANGE BRESTED BUNTING       0.88      0.85      0.86        52
             ORIENTAL BAY OWL       0.98      1.00      0.99        41
            ORNATE HAWK EAGLE       0.83      0.84      0.84        45
                       OSPREY       0.79      0.91      0.84        54
                      OSTRICH       0.87      0.94      0.91        36
                     OVENBIRD       0.98      0.87      0.92        67
               OYSTER CATCHER       0.98      0.96      0.97        53
              PAINTED BUNTING       0.96      0.90      0.93        49
                       PALILA       0.75      0.88      0.81        49
             PALM NUT VULTURE       0.92      0.84      0.88        55
             PARADISE TANAGER       0.96      0.98      0.97        48
              PARAKETT AUKLET       0.93      0.93      0.93        59
                  PARUS MAJOR       0.97      0.98      0.98        66
      PATAGONIAN SIERRA FINCH       0.88      0.88      0.88        57
                      PEACOCK       1.00      0.96      0.98        57
             PEREGRINE FALCON       0.79      0.80      0.79        55
                  PHAINOPEPLA       0.89      0.78      0.83        63
             PHILIPPINE EAGLE       0.97      0.97      0.97        68
                   PINK ROBIN       0.96      0.94      0.95        53
            PLUSH CRESTED JAY       0.90      0.95      0.93        60
              POMARINE JAEGER       0.83      0.70      0.76        54
                       PUFFIN       0.94      0.92      0.93        50
                    PUNA TEAL       0.69      0.85      0.76        39
                 PURPLE FINCH       0.90      0.81      0.85        53
             PURPLE GALLINULE       0.94      0.83      0.88        59
                PURPLE MARTIN       0.79      0.84      0.82        63
              PURPLE SWAMPHEN       0.86      0.88      0.87        50
             PYGMY KINGFISHER       0.83      0.89      0.86        54
                  PYRRHULOXIA       0.89      0.94      0.91        50
                      QUETZAL       0.92      0.92      0.92        59
             RAINBOW LORIKEET       1.00      1.00      1.00        58
                    RAZORBILL       0.87      0.89      0.88        62
        RED BEARDED BEE EATER       0.96      0.92      0.94        51
            RED BELLIED PITTA       0.98      0.95      0.97        63
        RED BILLED TROPICBIRD       0.80      0.91      0.85        58
             RED BROWED FINCH       0.92      0.91      0.92        54
                RED CROSSBILL       0.87      0.79      0.83        57
          RED FACED CORMORANT       0.83      0.81      0.82        64
            RED FACED WARBLER       0.98      0.94      0.96        65
                     RED FODY       0.94      0.85      0.89        52
              RED HEADED DUCK       0.89      0.91      0.90        56
        RED HEADED WOODPECKER       0.96      0.84      0.90        57
                     RED KNOT       0.90      0.85      0.87        61
      RED LEGGED HONEYCREEPER       0.96      0.89      0.92        55
             RED NAPED TROGON       0.89      0.92      0.91        53
          RED SHOULDERED HAWK       0.87      0.80      0.83        56
              RED TAILED HAWK       0.73      0.70      0.71        57
            RED TAILED THRUSH       0.95      0.98      0.97        60
         RED WINGED BLACKBIRD       0.86      0.76      0.81        67
          RED WISKERED BULBUL       0.93      0.90      0.92        61
             REGENT BOWERBIRD       0.87      0.96      0.91        48
         RING-NECKED PHEASANT       0.92      0.90      0.91        51
                   ROADRUNNER       0.90      0.96      0.93        49
                    ROCK DOVE       0.95      0.93      0.94        56
       ROSE BREASTED COCKATOO       0.90      0.92      0.91        51
       ROSE BREASTED GROSBEAK       0.98      0.91      0.94        54
            ROSEATE SPOONBILL       0.95      0.96      0.95        55
          ROSY FACED LOVEBIRD       0.91      0.94      0.93        54
            ROUGH LEG BUZZARD       0.64      0.71      0.67        49
             ROYAL FLYCATCHER       0.98      0.90      0.94        51
         RUBY CROWNED KINGLET       0.83      0.83      0.83        64
    RUBY THROATED HUMMINGBIRD       0.82      0.84      0.83        50
               RUDDY SHELDUCK       0.96      0.91      0.93        55
              RUDY KINGFISHER       0.97      0.98      0.97        59
            RUFOUS KINGFISHER       0.94      0.96      0.95        52
                 RUFOUS TREPE       0.91      0.91      0.91        47
                RUFUOS MOTMOT       0.96      0.91      0.94        58
              SAMATRAN THRUSH       0.86      0.89      0.87        54
                  SAND MARTIN       0.73      0.85      0.79        52
               SANDHILL CRANE       0.91      0.98      0.94        61
               SATYR TRAGOPAN       0.88      0.91      0.89        56
                  SAYS PHOEBE       0.84      0.84      0.84        69
   SCARLET CROWNED FRUIT DOVE       0.90      0.90      0.90        48
      SCARLET FACED LIOCICHLA       0.97      0.95      0.96        64
                 SCARLET IBIS       0.98      0.90      0.94        52
                SCARLET MACAW       0.98      0.96      0.97        54
              SCARLET TANAGER       0.92      0.92      0.92        50
                     SHOEBILL       0.88      0.92      0.90        53
       SHORT BILLED DOWITCHER       0.95      0.81      0.88        43
              SMITHS LONGSPUR       0.78      0.83      0.80        42
                   SNOW GOOSE       0.90      0.90      0.90        49
               SNOW PARTRIDGE       0.85      0.90      0.88        52
                  SNOWY EGRET       0.89      0.89      0.89        54
                    SNOWY OWL       0.91      0.89      0.90        55
                 SNOWY PLOVER       0.89      1.00      0.94        51
             SNOWY SHEATHBILL       0.83      0.88      0.85        57
                         SORA       0.84      0.93      0.88        44
             SPANGLED COTINGA       0.93      0.79      0.86        68
                SPLENDID WREN       0.96      0.88      0.92        51
        SPOON BILED SANDPIPER       0.93      0.93      0.93        56
              SPOTTED CATBIRD       0.82      0.83      0.82        64
       SPOTTED WHISTLING DUCK       0.85      0.88      0.87        52
                SQUACCO HERON       0.84      0.87      0.85        53
        SRI LANKA BLUE MAGPIE       1.00      0.96      0.98        46
                 STEAMER DUCK       0.88      0.89      0.88        47
      STORK BILLED KINGFISHER       0.95      0.90      0.92        58
            STRIATED CARACARA       0.85      0.84      0.84        55
                  STRIPED OWL       0.93      0.95      0.94        59
             STRIPPED MANAKIN       0.96      0.93      0.94        55
             STRIPPED SWALLOW       0.93      0.95      0.94        56
                   SUNBITTERN       0.83      0.87      0.85        63
              SUPERB STARLING       0.96      0.95      0.96        57
                  SURF SCOTER       0.88      0.95      0.92        40
            SWINHOES PHEASANT       0.89      0.87      0.88        54
                   TAILORBIRD       0.87      0.85      0.86        48
                TAIWAN MAGPIE       0.87      0.92      0.90        52
                       TAKAHE       0.96      0.94      0.95        47
                TASMANIAN HEN       0.88      0.92      0.90        49
              TAWNY FROGMOUTH       0.85      0.90      0.87        50
                    TEAL DUCK       0.90      0.65      0.75        54
                    TIT MOUSE       0.90      0.93      0.92        61
                      TOUCHAN       1.00      1.00      1.00        63
            TOWNSENDS WARBLER       0.84      0.92      0.88        50
                 TREE SWALLOW       0.87      0.76      0.81        59
         TRICOLORED BLACKBIRD       0.78      0.70      0.74        50
            TROPICAL KINGBIRD       0.76      0.82      0.79        51
                TRUMPTER SWAN       0.91      0.91      0.91        55
               TURKEY VULTURE       0.89      0.79      0.84        53
             TURQUOISE MOTMOT       0.95      0.94      0.94        62
                UMBRELLA BIRD       0.89      0.93      0.91        58
                VARIED THRUSH       0.90      0.94      0.92        49
                        VEERY       0.93      0.91      0.92        55
         VENEZUELIAN TROUPIAL       0.95      0.75      0.84        52
                       VERDIN       0.92      0.90      0.91        61
          VERMILION FLYCATHER       0.88      0.84      0.86        55
      VICTORIA CROWNED PIGEON       0.97      0.98      0.97        58
       VIOLET BACKED STARLING       0.93      0.91      0.92        44
                VIOLET CUCKOO       0.95      0.73      0.83        56
         VIOLET GREEN SWALLOW       0.85      0.85      0.85        54
                VIOLET TURACO       0.77      0.89      0.82        53
             VISAYAN HORNBILL       0.83      0.81      0.82        64
         VULTURINE GUINEAFOWL       0.89      0.98      0.94        59
                 WALL CREAPER       0.98      0.98      0.98        52
             WATTLED CURASSOW       0.84      0.93      0.88        41
              WATTLED LAPWING       0.98      0.95      0.97        59
                     WHIMBREL       0.87      0.96      0.91        55
      WHITE BREASTED WATERHEN       0.91      0.91      0.91        44
           WHITE BROWED CRAKE       0.91      0.83      0.86        58
         WHITE CHEEKED TURACO       0.72      0.83      0.77        47
       WHITE CRESTED HORNBILL       0.70      0.89      0.78        44
      WHITE EARED HUMMINGBIRD       0.84      0.81      0.82        52
           WHITE NECKED RAVEN       0.76      0.88      0.82        48
          WHITE TAILED TROPIC       0.87      0.79      0.83        61
     WHITE THROATED BEE EATER       0.92      0.94      0.93        52
                  WILD TURKEY       0.94      0.86      0.90        51
             WILLOW PTARMIGAN       0.88      0.91      0.90        57
     WILSONS BIRD OF PARADISE       0.96      0.94      0.95        51
                    WOOD DUCK       0.95      0.89      0.92        47
                  WOOD THRUSH       0.89      0.87      0.88        55
          WOODLAND KINGFISHER       0.91      0.93      0.92        45
                      WRENTIT       0.70      0.78      0.74        54
  YELLOW BELLIED FLOWERPECKER       0.88      0.88      0.88        56
         YELLOW BREASTED CHAT       0.91      0.82      0.86        62
               YELLOW CACIQUE       0.88      0.92      0.90        50
      YELLOW HEADED BLACKBIRD       0.88      0.92      0.90        49
                   ZEBRA DOVE       0.91      0.91      0.91        54

                     accuracy                           0.89     28665
                    macro avg       0.89      0.89      0.88     28665
                 weighted avg       0.89      0.89      0.89     28665

```  
# Conclusion and improvements  
 ResNet50 is a wonderful tool to aid in image classification. Some improvements to the model made here would be to use Max Pooling instead of Average Pooling. This is because Max pooling has the potential to better distinguish the foreground from the background.   
