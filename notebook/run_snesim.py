import pickle


def load_obj(dirname, name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'rb') as f:
        return pickle.load(f)


def save_obj(dirname, obj, name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


RunGeostatAlgorithm  snesim_std:
    :
        /GeostatParamUtils / XML:
            :
                <parameters > <algorithm name = "snesim_std" / > < Use_Affinity  value = "0" / > < Use_Rotation  value = "0" / > < Cmin  value = "1" / > < Constraint_Marginal_ADVANCED  value = "0" / > < resimulation_criterion  value = "-1" / > < resimulation_iteration_nb  value = "1" / > < Nb_Multigrids_ADVANCED  value = "5" / > < Debug_Level  value = "0" / > < Subgrid_choice  value = "0" / > < expand_isotropic  value = "1" / > < expand_anisotropic  value = "0" / > < aniso_factor  value = "        " / > < Hard_Data  grid = "lith_bin_rot_all" region = "" property = "Lith_binary" / > < use_pre_simulated_gridded_data  value = "0" / > < Use_ProbField  value = "0" / > < ProbField_properties count = "0"   value = "" / > < TauModelObject  value = "1 1" / > < use_vertical_proportion  value = "0" / > < GridSelector_Sim value = "swi_large_grid_2" region = "" / > < Property_Name_Sim  value = "snesim_fluv_smallTI" / > < Nb_Realizations  value = "1" / > < Seed  value = "211175" / > < PropertySelector_Training  grid = "swi_grid_small2" region = "" property = "ti_fluv__real0" / > < Nb_Facies  value = "2" / > < Marginal_Cdf  value = "0.33 0.67" / > < Max_Cond  value = "25" / > < Search_Ellipsoid  value = "1000 500 20  90 0 0" / > < /parameters >
