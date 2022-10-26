SEP_TOKEN = "<C>"

def createDataFrame(columnNames=[]):
    df = {}
    for colname in columnNames:
        df = createColumn(df, colname)
    return df

def createColumn(df, colname):
    df[colname] = []
    return df

def get_num_rows(df):
    currLen = 0
    for key in df.keys():
        currLen = max(currLen, len(df[key]))
    return currLen

def append_row(df, row_data):
    if isinstance(row_data, dict):
        return append_row_dict(df, row_data)
    return append_row_list(df, row_data)

def append_row_list(df, row_data):
    row_idx = get_num_rows(df)
    for i,key in enumerate(df.keys()):
        df[key].append("NA")
        df[key][row_idx] = row_data[i]
    return df

def append_row_dict(df, row_data):
    row_idx = get_num_rows(df)
    for key in df.keys():
        df[key].append("NA")
        if key in row_data.keys():
            df[key][row_idx] = row_data[key]
    return df

def read_row(df, row_idx):
    assert row_idx < get_num_rows(df)
    ret_list = []
    for key in df.keys():
        ret_list.append(df[key][row_idx])
    return ret_list

def print_df(df):
    outstr = to_str(df)
    outstr = outstr.replace("<C>", "\t")
    print(outstr)

def to_csv(df, fName, sep=","):
    outstr = to_str(df)
    outstr = outstr.replace("<C>", sep)
    text_file = open(fName, "w")
    n = text_file.write(outstr)
    text_file.close()

def to_str(df):
    outstr = ""
    col_names = df.keys()
    for colname in col_names:
        outstr+=str(colname) + SEP_TOKEN
    outstr = outstr[:-len(SEP_TOKEN)]
    outstr+="\n"
    for row_idx in range(get_num_rows(df)):
        row = read_row(df, row_idx)
        for val in row:
            outstr+=str(val) + SEP_TOKEN
        outstr = outstr[:-len(SEP_TOKEN)]
        outstr+="\n"
    return outstr