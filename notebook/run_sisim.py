import pickle


def load_obj(dirname, name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'rb') as f:
        return pickle.load(f)


def save_obj(dirname, obj, name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


varlist = load_obj(varlist)


cmd = ('RunGeostatAlgorithm  sisim::/GeostatParamUtils/XML::<parameters>  <algorithm name="sisim" />     <Hard_Data_Grid value="lith_bin_rot_all" region=""  />     ',
       '<Hard_Data_Property  value="Lith_binary"  />     <Assign_Hard_Data  value="1"  />     <coded_grid value="" region=""  />     <coded_props count="0"   value=""  />     ',
       '<Max_Conditioning_Data  value="12" />     <Search_Ellipsoid  value="1000 500 25  90 0 0" />    <AdvancedSearch  use_advanced_search="0"></AdvancedSearch>    '
       '<Variogram_Median_Ik  nugget="0" structures_count="1"  >    <structure_1  contribution="1"  type="Spherical"   >      <ranges max="500"  medium="300"  min="20"   />      '
       '<angles x="90"  y="0"  z="0"   />    </structure_1>  </Variogram_Median_Ik>    <Grid_Name value="swi_large_grid_2" region=""  />     <Property_Name  value="sisim" />     '
       '<Nb_Realizations  value="1" />     <Seed  value="14071789" />     <Categorical_Variable_Flag  value="1"  />     <Nb_Indicators  value="2" />     '
       '<Marginal_Probabilities  value="0.33 0.67" />     <lowerTailCdf  function ="Power"  extreme ="0"  omega ="3" />    <upperTailCdf  function ="Power"  extreme ="0"  omega ="0.333" />    '
       '<Median_Ik_Flag  value="1"  />     <Full_Ik_Flag  value="0"  />   </parameters>   ')
