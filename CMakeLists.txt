cmake_minimum_required(VERSION 3.16)
project(poisson2d CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- MPI ----
find_package(MPI REQUIRED)

add_library(common_objs OBJECT
  src/core.cpp
  src/main.cpp
)
target_include_directories(common_objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(common_objs PUBLIC MPI::MPI_CXX)

# ---- PETSc target ----
option(ENABLE_PETSC "Build PETSc backend" ON)
if (ENABLE_PETSC)
  # Prefer a proper FindPETSc if available
  find_package(PETSc QUIET)
  add_executable(poisson_petsc
    $<TARGET_OBJECTS:common_objs>
    src/petsc_solver.cpp
  )
  target_compile_definitions(poisson_petsc PRIVATE BACKEND_PETSC)
  target_include_directories(poisson_petsc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if (PETSc_FOUND)
    target_include_directories(poisson_petsc PRIVATE ${PETSC_INCLUDES})
    target_link_libraries(poisson_petsc PRIVATE ${PETSC_LIBRARIES})
  else()
    # Manual fallback via PETSC_DIR/PETSC_ARCH env or -D vars
    if (DEFINED PETSC_DIR AND DEFINED PETSC_ARCH)
      target_include_directories(poisson_petsc PRIVATE ${PETSC_DIR}/include ${PETSC_DIR}/${PETSC_ARCH}/include)
      
      target_link_directories(poisson_petsc PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/lib)
      target_link_libraries(poisson_petsc PRIVATE petsc)
    else()
      message(WARNING "PETSc not found. Set PETSC_DIR/PETSC_ARCH or provide FindPETSc.cmake.")
    endif()
  endif()
  target_link_libraries(poisson_petsc PRIVATE MPI::MPI_CXX)
endif()


# ---- HYPRE target ----
option(ENABLE_HYPRE "Build HYPRE backend" ON)
if (ENABLE_HYPRE)
  # Point CMake to hypre's package if needed, e.g. -DHYPRE_DIR=/home/costeas/hypre/install/lib/cmake/hypre
  find_package(HYPRE CONFIG REQUIRED)
  add_executable(poisson_hypre
    $<TARGET_OBJECTS:common_objs>
    src/hypre_solver.cpp
  )
  target_compile_definitions(poisson_hypre PRIVATE BACKEND_HYPRE)
  target_include_directories(poisson_hypre PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  # Link the imported target (gets the right includes, defs, transitive libs)
  target_link_libraries(poisson_hypre PRIVATE HYPRE::HYPRE MPI::MPI_CXX)
endif()

