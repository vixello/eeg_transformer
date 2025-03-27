# VS Code Setup

Create virtual environment  
`python -m venv venv`

Activate virtual environment  
`.\venv\Scripts\Activate`

Make sure that pip installation points to virtual environment  
`pip --version`

Install dependencies  
`pip install -r requirements.txt`

Install recommended extensions specified in  
`extensions.json`

# Scripts

### **download.py**

This script downloads raw datasets related to motor imagery. It accepts dataset name as a starting param.  
If none is provided, it will download all datasets. Supported dataset names are:  
`bci2a` - for BCI Competition IV 2a  
`bci2b` - for BCI Competition IV 2b  
`physionet` - for Physionet
