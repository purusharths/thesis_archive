include_directories(${PROJECT_BINARY_DIR}/src/include)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})
include_directories(graf/include)
add_subdirectory(graf/include/xtl)
# add_subdirectory(graf)
add_subdirectory(xtensor)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
# include_directories(PkgConfig:FFTW)
# \link_libraries(PkgConfig::FFTW)
if (WITH_ILUPP)
  include_directories(${ILUPP_INCLUDE_DIR})
  link_directories(${ILUPP_LIBRARY})
endif()

if (WITH_MUMPS)
  include_directories(${MUMPS_INCLUDE_DIR})
  link_directories(${MUMPS_LIBRARY})
  link_directories(${SMUMPS_LIBRARY})
  link_directories(${MUMPS_COMMON_LIBRARY})
  link_directories(${MUMPS_PORD_LIBRARY})
endif()

if (WITH_UMFPACK)
  include_directories(${UMFPACK_INCLUDE_DIR})
endif()

if(WITH_CBLAS)
  include_directories(${CBLAS_INCLUDE_DIR})
  link_directories(${CBLAS_LIBRARY})
endif()

if(WITH_CLAPACK)
  include_directories(${CBLAS_INCLUDE_DIR})
  include_directories(${CLAPACK_INCLUDE_DIR})
  link_directories(${CLAPACK_LIBRARIES})
  link_directories(${CBLAS_LIBRARY})
endif()

if (WITH_MKL)
  include_directories(${MKL_INCLUDE_DIR})
  link_directories(${MKL_LIBRARY})
endif()

if (WITH_ONEAPIMKL)
  include_directories(${ONEAPIMKL_INCLUDE_DIR})
  link_directories(${ONEAPIMKL_LIBRARY})
endif()

# HDF5
if (WITH_HDF5)
  include_directories(${HDF5_INCLUDE_DIR})
  link_directories(${HDF5_LIBRARY_DIRS})
endif()

if (WITH_LIKWID)
  include_directories(${LIKWID_INCLUDE_DIR})
  link_directories(${LIKWID_LIBRARY})
  add_definitions(-DLIKWID_PERFMON)
endif()

if (WITH_GPERF)
  include_directories(${GPERF_INCLUDE_DIR})
  link_directories(${GPERF_LIBRARY})
endif()

if (WITH_CUDA)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
endif()

if (WITH_HYPRE)
  include_directories(${HYPRE_INCLUDE_DIR})
  link_directories(${HYPRE_LIBRARIES})
endif()

if(WITH_OPENCL)
  include_directories(${OPENCL_INCLUDE_DIR})
endif()

if(WITH_PETSC)
  include_directories(${PETSC_INCLUDE_DIR})
endif()

if(WITH_COMPLEX_PETSC)
  include_directories(${PETSC_INCLUDE_DIR})
endif()

if(WITH_SLEPC)
  include_directories(${SLEPC_INCLUDE_DIR})
endif()

if (WITH_VCL)
  include_directories(${VCL_INCLUDE_DIR})
endif()

if (WITH_GRAPHCOLOR)
  include_directories(${GRAPHCOLOR_INCLUDE_DIR})
endif()

if (WITH_EXAFMM)
  include_directories(${EXAFMM_INCLUDE_DIR})
  link_directories(${EXAFMM_FFTW3_LIBRARY})
  link_directories(${EXAFMM_GFORTRAN_LIBRARY})
  link_directories(${EXAFMM_OPENBLAS_LIBRARY})
endif()

if(WITH_P4EST)
  include_directories(${P4EST_INCLUDE_DIR})
  link_directories(${P4EST_SC_LIBRARIES})
  link_directories(${P4EST_Z_LIBRARIES})
  link_directories(${P4EST_LIBRARIES})
endif()

add_definitions(-DMESHES_DATADIR="${PROJECT_BINARY_DIR}/exercises/data/")
add_definitions(-DHIFLOW_DIR="${PROJECT_SOURCE_DIR}")

set(SUBDIRS_EXERCISES
  data
  # ex0_poisson
  # ex1_poisson
  # fischerKPP
  #fischerKPP_epi
  #crypt_1_fischerKPP_epi
  #crypt2_fischerKPP_epi
  fischerKPP_GRAF_epi
  fischerKPP_para_epi
  fischerKPP_Parafields
  crypt1_fischerKPP_epi_BOTH
  crypt2FischerKPP_epi_BOTH
  #fischerKPP_single
  #graf_fischerKPP
  #para_fischerKPP
  fischerKPP_epi_BOTH
  #fischerKPP
  #fischerKPP_epithelial
  #ex2_NonlinearPoisson
  #ex3_CavityStokes
  #ex4_InstationaryStokes
  #ex5_StabilizedStokes
  #ex6_FoilStokes
  #ex7_CavityNavierStokes
  #ex8_FoilNavierStokes  
  #ex9_CavityNavierStokes
  #ex10_MagnusEffect
  #ex11_StokesConvergence
  #ex13_MagnusNaturalConvection
)

set(SUBDIRS_SOLUTION_EXERCISES
	#ex0_poisson
  	#ex1_poisson
  	#ex2_NonlinearPoisson
  	#ex3_CavityStokes
  	#ex4_InstationaryStokes
  	#ex5_StabilizedStokes
  	#ex6_FoilStokes
  	#ex7_CavityNavierStokes
  	#ex8_FoilNavierStokes  
  	#ex9_CavityNavierStokes
  	#ex10_MagnusEffect
  	#ex13_MagnusNaturalConvection
)

# Recurse through sub-directories and gather information for libhiflow
foreach(dir ${SUBDIRS_EXERCISES})

  include(${dir}/HiFlowFiles.cmake)

  make_directory(${PROJECT_BINARY_DIR}/exercises/${dir})

  foreach(i ${${dir}_SOURCES})
    add_executable(${i} ${dir}/${i}.cc)
    set_target_properties(${i} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${dir})
    target_link_libraries(${i} hiflow ${TINYXML2_LIBRARIES})
    if (WITH_HDF5)
      target_link_libraries(${i} pthread hdf5 dl)
    endif()
    if (WITH_GPERF)
      target_link_libraries(${i} profiler)
    endif()
    if(WITH_PETSC)
      target_link_libraries(${i} ${PETSC_LIBRARIES})
    endif()
    if (WITH_GPERF)
      target_link_libraries(${i} ${GPERF_LIBRARY})
    endif()
    if (WITH_EXAFMM)
      target_link_libraries(${i} gfortran )
      target_link_libraries(${i} openblas )
    endif()
    target_link_libraries(${i} xtl xtensor  PkgConfig::FFTW)

  endforeach()

  foreach(i ${${dir}_ADDITIONAL_FILES})
    file(COPY ${dir}/${i}
      DESTINATION ${PROJECT_BINARY_DIR}/exercises/${dir})
  endforeach()

  string(TOUPPER "${dir}" DIRECTORY)

endforeach()

# solution of exercises
foreach(dir ${SUBDIRS_SOLUTION_EXERCISES})

  include(${dir}/solution/HiFlowFiles.cmake)

  make_directory(${PROJECT_BINARY_DIR}/exercises/${dir}/solution)

  foreach(i ${${dir}_Solution_SOURCES})
    add_executable(${i}_sol ${dir}/solution/${i}.cc)
    set_target_properties(${i}_sol PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${dir}/solution)
    target_link_libraries(${i}_sol hiflow ${TINYXML2_LIBRARIES})
    if (WITH_HDF5)
      target_link_libraries(${i}_sol pthread hdf5 dl)
    endif()
    if (WITH_GPERF)
      target_link_libraries(${i}_sol profiler)
    endif()
    if(WITH_PETSC)
      target_link_libraries(${i}_sol ${PETSC_LIBRARIES})
    endif()
    if (WITH_GPERF)
      target_link_libraries(${i}_sol ${GPERF_LIBRARY})
    endif()
  endforeach()

  foreach(i ${${dir}_Solution_ADDITIONAL_FILES})
    file(COPY ${dir}/solution/${i}
      DESTINATION ${PROJECT_BINARY_DIR}/exercises/${dir}/solution)
  endforeach()

  string(TOUPPER "${dir}/solution" DIRECTORY)

endforeach()
