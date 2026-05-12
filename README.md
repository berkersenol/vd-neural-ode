VD Neural ODE -- Setup and Run Instructions
==========================================
Candidate: Berker Senol
Assignment: Vacuum Degasser Dynamic State Modelling


WHAT THIS CODE DOES
--------------------
Models the dynamic evolution of six state variables inside a vacuum degasser:
  dT/dt   -- temperature of the steel
  dp/dt   -- vessel pressure (pump characteristic curve)
  d[H]/dt -- hydrogen concentration
  d[N]/dt -- nitrogen concentration
  d[S]/dt -- sulfur concentration
  d[Al]/dt -- aluminium concentration

The model is trained as a Neural ODE using PyTorch and torchdiffeq.
It runs entirely on CPU -- no GPU required.


FOLDER STRUCTURE
-----------------
Put everything in one folder. The code finds all files relative to its
own location, so no path editing is needed.

  my_folder/
    vd_complete_assignment.py    <-- main code
    requirements.txt
    environment.yml
    README.md
    0_gr_-_nano_particle.xlsx    <-- data files (download separately)
    0_1_gr_-_nano_particle.xlsx
    0_3_gr_-_nano_particle.xlsx
    0_5_gr_-_nano_particle.xlsx
    outputs/                     <-- created automatically when code runs
      PLOT_A1_real_s45c.png
      PLOT_C4_coupled_ode.png
      PLOT_C6_sensitivity.png
      vd_synthetic_data.csv
      vd_model_weights.pt


SYSTEM REQUIREMENTS
--------------------
  Operating system : Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)
  Python version   : 3.10 or 3.11  (3.12 not yet tested with torchdiffeq)
  RAM              : 4 GB minimum, 8 GB recommended
  Disk space       : ~2 GB (mostly PyTorch)
  GPU              : Not required. Runs on CPU.


REQUIRED DATA FILES
--------------------
Download four Excel files from Riza and Ilman (2022).
Place them in the same folder as vd_complete_assignment.py.

  0_gr_-_nano_particle.xlsx      (pure water quench)
  0_1_gr_-_nano_particle.xlsx    (0.1g nanoparticle quench)
  0_3_gr_-_nano_particle.xlsx    (0.3g nanoparticle quench)
  0_5_gr_-_nano_particle.xlsx    (0.5g nanoparticle quench)

  Download from:
    DOI:     https://doi.org/10.1016/j.dib.2022.108867
    License: CC BY 4.0 (free to download and use)

If the files are not found the code skips Part A and continues with
Parts B and C. You will see a message per missing file:
  "0.0g pure water: file not found -- skipping"


OPTION 1 -- SETUP WITH PIP (recommended for most users)
---------------------------------------------------------
Step 1: Check your Python version
  python --version
  # Must show 3.10.x or 3.11.x

Step 2: Create a virtual environment
  # Windows:
  python -m venv vd_env
  vd_env\Scripts\activate

  # macOS / Linux:
  python -m venv vd_env
  source vd_env/bin/activate

  # You should see (vd_env) at the start of your terminal prompt.

Step 3: Install all packages
  pip install -r requirements.txt

  # Windows only -- if torch fails, run this first:
  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
  # Then re-run:
  pip install -r requirements.txt

Step 4: Run the code
  cd path/to/my_folder
  python vd_complete_assignment.py

Step 5: Check the outputs/ folder created in the same directory
  PLOT_A1_real_s45c.png    -- S45C real data with ODE fit
  PLOT_C4_coupled_ode.png  -- all 6 Neural ODE variables
  PLOT_C6_sensitivity.png  -- sensitivity analysis
  vd_synthetic_data.csv    -- generated training data
  vd_model_weights.pt      -- saved Neural ODE weights


OPTION 2 -- SETUP WITH CONDA (for Anaconda / Miniconda users)
--------------------------------------------------------------
Step 1: Create the environment
  conda env create -f environment.yml

Step 2: Activate
  conda activate vd_neural_ode

Step 3: Run
  cd path/to/my_folder
  python vd_complete_assignment.py

Step 4: Deactivate when finished
  conda deactivate


EXPECTED RUNTIME
----------------
  Part A -- S45C ODE fitting:      5 to 10 seconds
  Part B -- physics ODE solve:     2 to 5 seconds
  Part C1 -- synthetic data:       2 to 5 seconds
  Part C3 -- 1D Neural ODE:        1 to 3 minutes   (400 epochs)
  Part C4 -- 6D Neural ODE:        3 to 8 minutes   (400 epochs)
  Part C5 -- generalisation test:  10 to 30 seconds
  Part C6 -- sensitivity analysis: 10 to 30 seconds
  Total:                           5 to 12 minutes on a standard laptop CPU


COMMON ERRORS AND FIXES
------------------------
Error: ModuleNotFoundError: No module named 'torchdiffeq'
  Fix:  pip install torchdiffeq==0.2.3
        torch must be installed before torchdiffeq.

Error: ModuleNotFoundError: No module named 'torch'
  Fix:  pip install torch==2.4.1
        Windows: pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

Error: ModuleNotFoundError: No module named 'openpyxl'
  Fix:  pip install openpyxl==3.1.5
        openpyxl is required for pandas to read .xlsx files.

Error: "file not found -- skipping" messages for the xlsx files
  Not a crash. Download the files from DOI: 10.1016/j.dib.2022.108867
  and place them in the same folder as vd_complete_assignment.py.

Error: ValueError during ODE fitting in Part A
  Data quality issue in one Excel file.
  The code catches this per file and continues with the rest.

Error: RuntimeError: CUDA not available
  Should not happen -- the code selects CPU automatically.
  If it does: add  device = torch.device("cpu")  near the top of the file.


REFERENCES
----------
[1] Riza, R.I., Ilman, K.A. (2022). Data on the cooling rate using
    nano carbon-fluid quenching medium. Data in Brief, 46, 108867.
    DOI: 10.1016/j.dib.2022.108867 | CC BY 4.0

[2] Milijic, G. et al. (2024). Modelling of vacuum tank degassing.
    Process Metallurgy and Design of Furnaces, 1(4).
    DOI: 10.56578/pmdf010403 | CC BY 4.0

[3] Steneholm, K. et al. (2013). Removal of H, N, S from tool steel
    during vacuum degassing. Ironmaking and Steelmaking, 40, 199-205.
    DOI: 10.1179/1743281212Y.0000000029

[4] Vita, R. et al. (2024). Predicting VTD end-point temperature using ML.
    Processes, 12(7), 1414. DOI: 10.3390/pr12071414 | CC BY 4.0
