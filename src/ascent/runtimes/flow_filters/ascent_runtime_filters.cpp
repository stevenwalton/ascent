//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://ascent.readthedocs.io/.
// 
// Please also read ascent/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_filters.cpp
///
//-----------------------------------------------------------------------------

#include <ascent_runtime_filters.hpp>
#include <ascent_main_runtime.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <flow_workspace.hpp>

#include <ascent_runtime_relay_filters.hpp>
#include <ascent_runtime_blueprint_filters.hpp>

#if defined(ASCENT_VTKM_ENABLED)
    #include <ascent_runtime_vtkh_filters.hpp>
#endif

#ifdef ASCENT_MPI_ENABLED
    #include <ascent_runtime_hola_filters.hpp>
#if defined(ASCENT_ADIOS_ENABLED)
    #include <ascent_runtime_adios_filters.hpp>
#endif // ADIOS

#if defined(ASCENT_ADIOS2_ENABLED)
    #include <ascent_runtime_adios2_filters.hpp>
#endif // ADIOS2

#endif // MPI



using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{


//-----------------------------------------------------------------------------
// init all built in filters
//-----------------------------------------------------------------------------
void
register_builtin()
{
    AscentRuntime::register_filter_type<BlueprintVerify>(); 
    AscentRuntime::register_filter_type<EnsureLowOrder>(); 
    AscentRuntime::register_filter_type<EnsureBlueprint>();
    AscentRuntime::register_filter_type<RelayIOSave>("extracts","relay");
    AscentRuntime::register_filter_type<RelayIOLoad>();
    
#if defined(ASCENT_VTKM_ENABLED)
    AscentRuntime::register_filter_type<DefaultRender>();
    AscentRuntime::register_filter_type<EnsureVTKH>(); 
    AscentRuntime::register_filter_type<EnsureVTKM>();
    // TODO: This should be moved to blueprint filters 

    AscentRuntime::register_filter_type<VTKHBounds>();
    AscentRuntime::register_filter_type<VTKHUnionBounds>();

    AscentRuntime::register_filter_type<VTKHDomainIds>();
    AscentRuntime::register_filter_type<VTKHUnionDomainIds>();
    
    AscentRuntime::register_filter_type<DefaultScene>();

    // transforms, the current crop expect vtk-h input data
    AscentRuntime::register_filter_type<VTKHClip>("transforms","clip");
    AscentRuntime::register_filter_type<VTKHClipWithField>("transforms","clip_with_field");
    AscentRuntime::register_filter_type<VTKHIsoVolume>("transforms","isovolume");
    AscentRuntime::register_filter_type<VTKHMarchingCubes>("transforms","contour");
    AscentRuntime::register_filter_type<VTKHThreshold>("transforms","threshold");
    AscentRuntime::register_filter_type<VTKHSlice>("transforms","slice");
    AscentRuntime::register_filter_type<VTKH3Slice>("transforms","3slice");
    AscentRuntime::register_filter_type<VTKHNoOp>("transforms","noop");

    AscentRuntime::register_filter_type<AddPlot>();
    AscentRuntime::register_filter_type<CreatePlot>();
    AscentRuntime::register_filter_type<CreateScene>();
    AscentRuntime::register_filter_type<ExecScene>();
#endif // VTKM

#if defined(ASCENT_MPI_ENABLED)
    AscentRuntime::register_filter_type<HolaMPIExtract>("extracts","hola_mpi");

#if defined(ASCENT_ADIOS_ENABLED)
    AscentRuntime::register_filter_type<ADIOS>("extracts","adios");
#endif // ADIOS
#if defined(ASCENT_ADIOS2_ENABLED)
    AscentRuntime::register_filter_type<ASCENT_ADIOS2>("extracts","adios2");
#endif // ADIOS2

#endif // MPI

}



//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

