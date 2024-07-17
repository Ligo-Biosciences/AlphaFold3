"""Ligand constants used in AlphaFold3."""

import collections
import functools
import os
from typing import Final, List, Mapping, Tuple, Set

import numpy as np
import tree

# Ligand exclusion list.
LIGAND_EXCLUSION_LIST: Final[Set[str]] = {
    "144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE", "AAE", "ABA", "ACE", "ACN", "ACT",
    "ACY", "AZI", "BAM", "BCN", "BCT", "BDN", "BEN", "BME", "BO3", "BTB", "BTC", "BU1", "C8E", "CAD", "CAQ",
    "CBM", "CCN", "CIT", "CL", "CLR", "CM", "CMO", "CO3", "CPT", "CXS", "D10", "DEP", "DIO", "DMS", "DN",
    "DOD", "DOX", "EDO", "EEE", "EGL", "EOH", "EOX", "EPE", "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL",
    "GSH", "GTT", "GYF", "HED", "IHP", "IHS", "IMD", "IOD", "IPA", "IPH", "LDA", "MB3", "MEG", "MES", "MLA",
    "MLI", "MOH", "MPD", "MRD", "MSE", "MYR", "N", "NA", "NH2", "NH4", "NHE", "NO3", "O4B", "OHE", "OLA", "OLC",
    "OMB", "OME", "OXA", "P6G", "PE3", "PE4", "PEG", "PEO", "PEP", "PG0", "PG4", "PGE", "PGR", "PLM", "PO4",
    "POL", "POP", "PVO", "SAR", "SCN", "SEO", "SEP", "SIN", "SO4", "SPD", "SPM", "SR", "STE", "STO", "STU",
    "TAR", "TBU", "TME", "TPO", "TRS", "UNK", "UNL", "UNX", "UPL", "URE"
}

# Crystallization aids
CRYSTALLIZATION_AIDS: Final[Set[str]] = {
    "SO4", "GOL", "EDO", "PO4", "ACT", "PEG", "DMS", "TRS", "PGE", "PG4", "FMT", "EPE", "MPD", "MES", "CD", "IOD"
}

# Ions
IONS: Final[Set[str]] = {
    '118', '119', '1AL', '1CU', '2FK', '2HP', '2OF', '3CO', '3MT', '3NI', '3OF', '4MO', '4PU', '4TI', '543',
    '6MO', 'AG', 'AL', 'ALF', 'AM', 'ATH', 'AU', 'AU3', 'AUC', 'BA', 'BEF', 'BF4', 'BO4', 'BR', 'BS3', 'BSY',
    'CA', 'CAC', 'CD', 'CD1', 'CD3', 'CD5', 'CE', 'CF', 'CHT', 'CO', 'CO5', 'CON', 'CR', 'CS', 'CSB', 'CU',
    'CU1', 'CU2', 'CU3', 'CUA', 'CUZ', 'CYN', 'DME', 'DMI', 'DSC', 'DTI', 'DY', 'E4N', 'EDR', 'EMC', 'ER3',
    'EU', 'EU3', 'F', 'FE', 'FE2', 'FPO', 'GA', 'GD3', 'GEP', 'HAI', 'HG', 'HGC', 'HO3', 'IN', 'IR', 'IR3',
    'IRI', 'IUM', 'K', 'KO4', 'LA', 'LCO', 'LCP', 'LI', 'LU', 'MAC', 'MG', 'MH2', 'MH3', 'MMC', 'MN', 'MN3',
    'MN5', 'MN6', 'MO', 'MO1', 'MO2', 'MO3', 'MO4', 'MO5', 'MO6', 'MOO', 'MOS', 'MOW', 'MW1', 'MW2', 'MW3',
    'NA2', 'NA5', 'NA6', 'NAO', 'NAW', 'NET', 'NI', 'NI1', 'NI2', 'NI3', 'NO2', 'NRU', 'O4M', 'OAA', 'OC1',
    'OC2', 'OC3', 'OC4', 'OC5', 'OC6', 'OC7', 'OC8', 'OCL', 'OCM', 'OCN', 'OCO', 'OF1', 'OF2', 'OF3', 'OH',
    'OS', 'OS4', 'OXL', 'PB', 'PBM', 'PD', 'PER', 'PI', 'PO3', 'PR', 'PT', 'PT4', 'PTN', 'RB', 'RH3', 'RHD',
    'RU', 'SB', 'SE4', 'SEK', 'SM', 'SMO', 'SO3', 'T1A', 'TB', 'TBA', 'TCN', 'TEA', 'TH', 'THE', 'TL', 'TMA',
    'TRA', 'V', 'VN3', 'VO4', 'W', 'WO5', 'Y1', 'YB', 'YB2', 'YH', 'YT3', 'ZCM', 'ZN', 'ZN2', 'ZN3', 'ZNO',
    'ZO3', 'ZR'
}

# Glycan CCD codes
GLYCAN_CODES: Final[Set[str]] = {
    '045', '05L', '07E', '07Y', '08U', '09X', '0BD', '0H0', '0HX', '0LP', '0MK', '0NZ', '0UB', '0V4', '0WK',
    '0XY', '0YT', '10M', '12E', '145', '147', '149', '14T', '15L', '16F', '16G', '16O', '17T', '18D', '18O',
    '1CF', '1FT', '1GL', '1GN', '1LL', '1S3', '1S4', '1SD', '1X4', '20S', '20X', '22O', '22S', '23V', '24S',
    '25E', '26O', '27C', '289', '291', '293', '2DG', '2DR', '2F8', '2FG', '2FL', '2GL', '2GS', '2H5', '2HA',
    '2M4', '2M5', '2M8', '2OS', '2WP', '2WS', '32O', '34V', '38J', '3BU', '3DO', '3DY', '3FM', '3GR', '3HD',
    '3J3', '3J4', '3LJ', '3LR', '3MG', '3MK', '3R3', '3S6', '3SA', '3YW', '40J', '42D', '445', '44S', '46D',
    '46Z', '475', '48Z', '491', '49A', '49S', '49T', '49V', '4AM', '4CQ', '4GC', '4GL', '4GP', '4JA', '4N2',
    '4NN', '4QY', '4R1', '4RS', '4SG', '4UZ', '4V5', '50A', '51N', '56N', '57S', '5GF', '5GO', '5II', '5KQ',
    '5KS', '5KT', '5KV', '5L3', '5LS', '5LT', '5MM', '5N6', '5QP', '5SP', '5TH', '5TJ', '5TK', '5TM', '61J',
    '62I', '64K', '66O', '6BG', '6C2', '6DM', '6GB', '6GP', '6GR', '6K3', '6KH', '6KL', '6KS', '6KU', '6KW',
    '6LA', '6LS', '6LW', '6MJ', '6MN', '6PZ', '6S2', '6UD', '6YR', '6ZC', '73E', '79J', '7CV', '7D1', '7GP',
    '7JZ', '7K2', '7K3', '7NU', '83Y', '89Y', '8B7', '8B9', '8EX', '8GA', '8GG', '8GP', '8I4', '8LR', '8OQ',
    '8PK', '8S0', '8YV', '95Z', '96O', '98U', '9AM', '9C1', '9CD', '9GP', '9KJ', '9MR', '9OK', '9PG', '9QG',
    '9S7', '9SG', '9SJ', '9SM', '9SP', '9T1', '9T7', '9VP', '9WJ', '9WN', '9WZ', '9YW', 'A0K', 'A1Q', 'A2G',
    'A5C', 'A6P', 'AAL', 'ABD', 'ABE', 'ABF', 'ABL', 'AC1', 'ACR', 'ACX', 'ADA', 'AF1', 'AFD', 'AFO', 'AFP',
    'AGL', 'AH2', 'AH8', 'AHG', 'AHM', 'AHR', 'AIG', 'ALL', 'ALX', 'AMG', 'AMN', 'AMU', 'AMV', 'ANA', 'AOG',
    'AQA', 'ARA', 'ARB', 'ARI', 'ARW', 'ASC', 'ASG', 'ASO', 'AXP', 'AXR', 'AY9', 'AZC', 'B0D', 'B16', 'B1H',
    'B1N', 'B2G', 'B4G', 'B6D', 'B7G', 'B8D', 'B9D', 'BBK', 'BBV', 'BCD', 'BDF', 'BDG', 'BDP', 'BDR', 'BEM',
    'BFN', 'BG6', 'BG8', 'BGC', 'BGL', 'BGN', 'BGP', 'BGS', 'BHG', 'BM3', 'BM7', 'BMA', 'BMX', 'BND', 'BNG',
    'BNX', 'BO1', 'BOG', 'BQY', 'BS7', 'BTG', 'BTU', 'BW3', 'BWG', 'BXF', 'BXP', 'BXX', 'BXY', 'BZD', 'C3B',
    'C3G', 'C3X', 'C4B', 'C4W', 'C5X', 'CBF', 'CBI', 'CBK', 'CDR', 'CE5', 'CE6', 'CE8', 'CEG', 'CEZ', 'CGF',
    'CJB', 'CKB', 'CKP', 'CNP', 'CR1', 'CR6', 'CRA', 'CT3', 'CTO', 'CTR', 'CTT', 'D1M', 'D5E', 'D6G', 'DAF',
    'DAG', 'DAN', 'DDA', 'DDL', 'DEG', 'DEL', 'DFR', 'DFX', 'DG0', 'DGO', 'DGS', 'DGU', 'DJB', 'DJE', 'DK4',
    'DKX', 'DKZ', 'DL6', 'DLD', 'DLF', 'DLG', 'DNO', 'DO8', 'DOM', 'DPC', 'DQR', 'DR2', 'DR3', 'DR5', 'DRI',
    'DSR', 'DT6', 'DVC', 'DYM', 'E3M', 'E5G', 'EAG', 'EBG', 'EBQ', 'EEN', 'EEQ', 'EGA', 'EMP', 'EMZ', 'EPG',
    'EQP', 'EQV', 'ERE', 'ERI', 'ETT', 'EUS', 'F1P', 'F1X', 'F55', 'F58', 'F6P', 'F8X', 'FBP', 'FCA', 'FCB',
    'FCT', 'FDP', 'FDQ', 'FFC', 'FFX', 'FIF', 'FK9', 'FKD', 'FMF', 'FMO', 'FNG', 'FNY', 'FRU', 'FSA', 'FSI',
    'FSM', 'FSW', 'FUB', 'FUC', 'FUD', 'FUF', 'FUL', 'FUY', 'FVQ', 'FX1', 'FYJ', 'G0S', 'G16', 'G1P', 'G20',
    'G28', 'G2F', 'G3F', 'G3I', 'G4D', 'G4S', 'G6D', 'G6P', 'G6S', 'G7P', 'G8Z', 'GAA', 'GAC', 'GAD', 'GAF',
    'GAL', 'GAT', 'GBH', 'GC1', 'GC4', 'GC9', 'GCB', 'GCD', 'GCN', 'GCO', 'GCS', 'GCT', 'GCU', 'GCV', 'GCW',
    'GDA', 'GDL', 'GE1', 'GE3', 'GFP', 'GIV', 'GL0', 'GL1', 'GL2', 'GL4', 'GL5', 'GL6', 'GL7', 'GL9', 'GLA',
    'GLC', 'GLD', 'GLF', 'GLG', 'GLO', 'GLP', 'GLS', 'GLT', 'GM0', 'GMB', 'GMH', 'GMT', 'GMZ', 'GN1', 'GN4',
    'GNS', 'GNX', 'GP0', 'GP1', 'GP4', 'GPH', 'GPK', 'GPM', 'GPO', 'GPQ', 'GPU', 'GPV', 'GPW', 'GQ1', 'GRF',
    'GRX', 'GS1', 'GS9', 'GTK', 'GTM', 'GTR', 'GU0', 'GU1', 'GU2', 'GU3', 'GU4', 'GU5', 'GU6', 'GU8', 'GU9',
    'GUF', 'GUL', 'GUP', 'GUZ', 'GXL', 'GXV', 'GYE', 'GYG', 'GYP', 'GYU', 'GYV', 'GZL', 'H1M', 'H1S', 'H2P',
    'H3S', 'H53', 'H6Q', 'H6Z', 'HBZ', 'HD4', 'HNV', 'HNW', 'HSG', 'HSH', 'HSJ', 'HSQ', 'HSX', 'HSY', 'HTG',
    'HTM', 'HVC', 'IAB', 'IDC', 'IDF', 'IDG', 'IDR', 'IDS', 'IDU', 'IDX', 'IDY', 'IEM', 'IN1', 'IPT', 'ISD',
    'ISL', 'ISX', 'IXD', 'J5B', 'JFZ', 'JHM', 'JLT', 'JRV', 'JSV', 'JV4', 'JVA', 'JVS', 'JZR', 'K5B', 'K99',
    'KBA', 'KBG', 'KD5', 'KDA', 'KDB', 'KDD', 'KDE', 'KDF', 'KDM', 'KDN', 'KDO', 'KDR', 'KFN', 'KG1', 'KGM',
    'KHP', 'KME', 'KO1', 'KO2', 'KOT', 'KTU', 'L0W', 'L1L', 'L6S', 'L6T', 'LAG', 'LAH', 'LAI', 'LAK', 'LAO',
    'LAT', 'LB2', 'LBS', 'LBT', 'LCN', 'LDY', 'LEC', 'LER', 'LFC', 'LFR', 'LGC', 'LGU', 'LKA', 'LKS', 'LM2',
    'LMO', 'LNV', 'LOG', 'LOX', 'LRH', 'LTG', 'LVO', 'LVZ', 'LXB', 'LXC', 'LXZ', 'LZ0', 'M1F', 'M1P', 'M2F',
    'M3M', 'M3N', 'M55', 'M6D', 'M6P', 'M7B', 'M7P', 'M8C', 'MA1', 'MA2', 'MA3', 'MA8', 'MAB', 'MAF', 'MAG',
    'MAL', 'MAN', 'MAT', 'MAV', 'MAW', 'MBE', 'MBF', 'MBG', 'MCU', 'MDA', 'MDP', 'MFB', 'MFU', 'MG5', 'MGC',
    'MGL', 'MGS', 'MJJ', 'MLB', 'MLR', 'MMA', 'MN0', 'MNA', 'MQG', 'MQT', 'MRH', 'MRP', 'MSX', 'MTT', 'MUB',
    'MUR', 'MVP', 'MXY', 'MXZ', 'MYG', 'N1L', 'N3U', 'N9S', 'NA1', 'NAA', 'NAG', 'NBG', 'NBX', 'NBY', 'NDG',
    'NFG', 'NG1', 'NG6', 'NGA', 'NGC', 'NGE', 'NGK', 'NGR', 'NGS', 'NGY', 'NGZ', 'NHF', 'NLC', 'NM6', 'NM9',
    'NNG', 'NPF', 'NSQ', 'NT1', 'NTF', 'NTO', 'NTP', 'NXD', 'NYT', 'OAK', 'OI7', 'OPM', 'OSU', 'OTG', 'OTN',
    'OTU', 'OX2', 'P53', 'P6P', 'P8E', 'PA1', 'PAV', 'PDX', 'PH5', 'PKM', 'PNA', 'PNG', 'PNJ', 'PNW', 'PPC',
    'PRP', 'PSG', 'PSV', 'PTQ', 'PUF', 'PZU', 'QDK', 'QIF', 'QKH', 'QPS', 'QV4', 'R1P', 'R1X', 'R2B', 'R2G',
    'RAE', 'RAF', 'RAM', 'RAO', 'RB5', 'RBL', 'RCD', 'RER', 'RF5', 'RG1', 'RGG', 'RHA', 'RHC', 'RI2', 'RIB',
    'RIP', 'RM4', 'RP3', 'RP5', 'RP6', 'RR7', 'RRJ', 'RRY', 'RST', 'RTG', 'RTV', 'RUG', 'RUU', 'RV7', 'RVG',
    'RVM', 'RWI', 'RY7', 'RZM', 'S7P', 'S81', 'SA0', 'SCG', 'SCR', 'SDY', 'SEJ', 'SF6', 'SF9', 'SFU', 'SG4',
    'SG5', 'SG6', 'SG7', 'SGA', 'SGC', 'SGD', 'SGN', 'SHB', 'SHD', 'SHG', 'SIA', 'SID', 'SIO', 'SIZ', 'SLB',
    'SLM', 'SLT', 'SMD', 'SN5', 'SNG', 'SOE', 'SOG', 'SOL', 'SOR', 'SR1', 'SSG', 'SSH', 'STW', 'STZ', 'SUC',
    'SUP', 'SUS', 'SWE', 'SZZ', 'T68', 'T6D', 'T6P', 'T6T', 'TA6', 'TAG', 'TCB', 'TDG', 'TEU', 'TF0', 'TFU',
    'TGA', 'TGK', 'TGR', 'TGY', 'TH1', 'TM5', 'TM6', 'TMR', 'TMX', 'TNX', 'TOA', 'TOC', 'TQY', 'TRE', 'TRV',
    'TS8', 'TT7', 'TTV', 'TU4', 'TUG', 'TUJ', 'TUP', 'TUR', 'TVD', 'TVG', 'TVM', 'TVS', 'TVV', 'TVY', 'TW7',
    'TWA', 'TWD', 'TWG', 'TWJ', 'TWY', 'TXB', 'TYV', 'U1Y', 'U2A', 'U2D', 'U63', 'U8V', 'U97', 'U9A', 'U9D',
    'U9G', 'U9J', 'U9M', 'UAP', 'UBH', 'UBO', 'UDC', 'UEA', 'V3M', 'V3P', 'V71', 'VG1', 'VJ1', 'VJ4', 'VKN',
    'VTB', 'W9T', 'WIA', 'WOO', 'WUN', 'WZ1', 'WZ2', 'X0X', 'X1P', 'X1X', 'X2F', 'X2Y', 'X34', 'X6X', 'X6Y',
    'XDX', 'XGP', 'XIL', 'XKJ', 'XLF', 'XLS', 'XMM', 'XS2', 'XXM', 'XXR', 'XXX', 'XYF', 'XYL', 'XYP', 'XYS',
    'XYT', 'XYZ', 'YDR', 'YIO', 'YJM', 'YKR', 'YO5', 'YX0', 'YX1', 'YYB', 'YYH', 'YYJ', 'YYK', 'YYM', 'YYQ',
    'YZ0', 'Z0F', 'Z15', 'Z16', 'Z2D', 'Z2T', 'Z3K', 'Z3L', 'Z3Q', 'Z3U', 'Z4K', 'Z4R', 'Z4S', 'Z4U', 'Z4V',
    'Z4W', 'Z4Y', 'Z57', 'Z5J', 'Z5L', 'Z61', 'Z6H', 'Z6J', 'Z6W', 'Z8H', 'Z8T', 'Z9D', 'Z9E', 'Z9H', 'Z9K',
    'Z9L', 'Z9M', 'Z9N', 'Z9W', 'ZB0', 'ZB1', 'ZB2', 'ZB3', 'ZCD', 'ZCZ', 'ZD0', 'ZDC', 'ZDO', 'ZEE', 'ZEL',
    'ZGE', 'ZMR '
}

