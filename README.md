<div align="center">
    
# Transcatheter Heart Valve Parametrization System

</div>

This repository is linked to the study "Towards Automatic Transcatheter Heart Valve Parametrisation on X-ray Sequences" (see report.pdf for details).  It contains a computational framework for parametrising the contours of transcatheter heart valves from X-ray image sequences.



<div align="center">
    
![](https://github.com/AlmaVivasLago/thv-model/assets/64702159/86e2851e-2728-43a0-85f5-f2a9bb46a197)

</div>









## Requirements

Python 3.8 or higher.

## Installation

To set up the system, clone the repository and install the dependencies as follows:

```bash
git clone https://github.com/AlmaVivasLago/thv-model.git
cd thv-model
pip install -r requirements.txt
```

## Repository Structure
After cloning the repository, you'll have the following directory layout:

```
thv-model/
├── README.md
├── requirements.txt
├── report.pdf
├── src/
│   ├── main.py
│   └── (additional source files)
└── data/
    ├── A/
    │   ├── A0/
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── A1/
    │   └── A2/
    ├── B/
    └── (additional patient folders)
```
        

 **Note:** The `data` directory is an example structure only. Actual patient data is not included due to privacy concerns.


## Usage
To run the THV parametrization process, use the following command:

```
python src/main.py --path <DATA_DIRECTORY_PATH>
```

## Expected Results
Upon running the parametrization process, the system will generate the following results:

- extrema_coordinates.npy: contains a vector of the contour coordinates representing the parametrized transcatheter heart valve (THV) geometry.
  
-  Annotated Frames:  individual frames from the X-ray sequence will be provided, each annotated with the parametrized THV contour. 

- overplot_sequence.mp4: alongside the frames, a video file overplotting the parametrized THV contour onto the original X-ray image sequence.


## Additional Information

Faster, partial C++ implementation available for potential real-time applications (status: stopped). Reach out if interested.


