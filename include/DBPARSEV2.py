import pandas as pd
from datetime import datetime, timedelta

def Function_READBEAM(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['Time Start'] = pd.to_datetime(df['Time Start'].str.strip())
    df['Time End'] = pd.to_datetime(df['Time End'].str.strip())
    time_start = df['Time Start'].values
    time_end = df['Time End'].values
    polarization = df['Polarization'].values
    error = df['Error'].values    
    return time_start, time_end, polarization, error

def Function_HE3POL(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['Time'].str.strip())
    time = df['Time'].values
    polarization = df['Polarization'].values
    return time, polarization

def Function_READRUNLIST(file_path):
    df = pd.read_csv(file_path, header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df['Start'] = pd.to_datetime(df['Start'].str.strip(), errors='coerce')
    df['Finish'] = pd.to_datetime(df['Finish'].str.strip(), errors='coerce')
    
    df['Finish'] = df.apply(
        lambda row: row['Finish'] + timedelta(days=1) if row['Finish'] < row['Start'] else row['Finish'],
        axis=1
    )
    return df

def Function_RETURNHE3POL(runnum, time_file_path, pol_file_path):
    """
    Returns the average polarization for the input run number during the duration of the run.
    :param runnum: The run number to lookup.
    :param time_file_path: Path to the runlist CSV file.
    :param pol_file_path: Path to the polarization CSV file.
    :return: Average polarization during the run.
    """
    runlist_df = Function_READRUNLIST(time_file_path)
    run_data = runlist_df[runlist_df['Run'] == runnum]
    if run_data.empty:
        raise ValueError(f"Run number {runnum} not found in runlist.")
    start_time = run_data['Start'].values[0]
    end_time = run_data['Finish'].values[0]
    time, polarization = Function_HE3POL(pol_file_path)
    mask = (time >= start_time) & (time <= end_time)
    filtered_polarization = polarization[mask]    
    if len(filtered_polarization) == 0:
        raise ValueError(f"No polarization data found for run number {runnum} within the run duration.")
    average_polarization = filtered_polarization.mean()
    return average_polarization

def Function_RETURNBEAMPOL(runnum, time_file_path, beam_pol_file_path):
    """
    Returns the average beam polarization for the input run number during the duration of the run.

    :param runnum: The run number to lookup.
    :param time_file_path: Path to the runlist CSV file.
    :param beam_pol_file_path: Path to the beam polarization CSV file.
    :return: Beam polarization during the run.
    """
    # Read the runlist CSV
    runlist_df = Function_READRUNLIST(time_file_path)
    
    # Get the start and end time for the given run number
    run_data = runlist_df[runlist_df['Run'] == runnum]
    if run_data.empty:
        raise ValueError(f"Run number {runnum} not found in runlist.")
    
    start_time = run_data['Start'].values[0]
    end_time = run_data['Finish'].values[0]
    
    # Read the beam polarization data
    time_start, time_end, polarization, error = Function_READBEAM(beam_pol_file_path)
    
    # Find the beam polarization that covers the run time
    for i in range(len(time_start)):
        if start_time >= time_start[i] and end_time <= time_end[i]:
            return polarization[i]
    
    raise ValueError(f"No beam polarization data found for run number {runnum} within the run duration.")


def Function_RETURNPROCESSEDBEAMPOL(runnum):    
    import pandas as pd
    data = pd.read_csv("../DB/ProcessedBeamPol.csv")
    polarization_value = data.loc[data['Run Number'] == runnum, 'Polarization'].values[0]
    return polarization_value
def Function_RETURNPROCESSEDHE3POL(runnum):    
    import pandas as pd
    data = pd.read_csv("../DB/ProcessedHe3Pol.csv")
    polarization_value = data.loc[data['Run Number'] == runnum, 'Polarization'].values[0]
    return polarization_value

    

    