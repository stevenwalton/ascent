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
/// file: ascent_runtime_adios_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_FLOW_PIPELINE_ADIOS2_FILTERS_HPP
#define ASCENT_FLOW_PIPELINE_ADIOS2_FILTERS_HPP

#include <flow_filter.hpp>
#include <adios2.h>
#include <memory>
#include <ascent_logging.hpp>
#include <map>

#ifdef ADIOS2_HAVE_MPI
#include <mpi.h>
#endif


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
///
/// Filters Related to Conduit Relay IO
///
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/*
 *  Writer manager is static so that it stays around after ASCENT_ADIOS2 
 *  constructs freqently
 */
class WriterManager//: public ::flow::Filter
{
    public:
        WriterManager(std::string file_name, 
                      std::string engine_name, 
                      std::string ip_address, 
                      std::string port_number,
                      std::string Marshal_Method);
        WriterManager();
        ~WriterManager();

    	bool UniformMeshSchema(const conduit::Node &node);

    	bool RectilinearMeshSchema(const conduit::Node &node);
        bool m_recFirst = true;

    	bool ExplicitMeshSchema(const conduit::Node &node);
        bool m_explicitMeshFirst = true;

    	bool FieldVariable(const std::string &fieldName, 
			   const conduit::Node &fieldNode, 
			   const conduit::Node &n_topo);
        bool m_fieldVarFirst = true;

    	bool CalcRectilinearMeshInfo(const conduit::Node &node,
    	                             std::vector<std::vector<double>> &globalCoords);

    	bool CalcExplicitMeshInfo(const conduit::Node &node,
				  std::vector<std::vector<double>> &globalCoords);

        int rank, numRanks;
        bool m_explicitMesh;
        adios2::Engine m_adiosWriter;
        bool firstStep = true;
        std::string m_engineName;
    protected:
        std::shared_ptr<adios2::ADIOS> adios;
        static std::map<std::string, adios2::Engine> m_writer;
        std::vector<adios2::Variable<double>> myVars;
        adios2::IO m_IO;

        std::string m_fileName;
        std::string m_meshName;
        std::string m_ip;
        std::string m_port;
        std::string m_marshal;

        //var dimensions for this rank:
        std::vector<std::int64_t> m_globalDims, m_localDims, m_offset;
        std::map<std::string,adios2::Variable<double>> m_fieldVariableMap;


    private:
#ifdef ADIOS2_HAVE_MPI
        MPI_Comm mpi_comm;
#else
        int mpi_comm;
#endif

        // Determine is a variable is point centered or cell centered
        template <typename T>
        std::vector<std::size_t> pointOrCell(const std::vector<T> &data, bool pointCentered=true, bool explicitMesh=false)
        {
            std::size_t numRows = data.size();
            // for vector data
            if ( numRows > 1 )
            {
                // Set a vector of correct type to values of data
                std::vector<std::size_t> returnBlock(data.begin(), data.end());
                for ( std::size_t i = 0; i < numRows; ++i )
                {
                    // If point data -> reduce the dims by 1
                    if ( (i > 0 || explicitMesh ) && !pointCentered && returnBlock[i] > 0 )
                        returnBlock[i]--;
                }
                return returnBlock;
            }
            // Scalar data
            else if ( numRows == 1 )
            {
                std::vector<std::size_t> returnBlock(1, data[data.size()-1]);
                if ( !pointCentered && returnBlock[0] > 0 )
                    returnBlock[0]--;
                return returnBlock;
            }
            // Error data!!
            ASCENT_ERROR("The data you are passing the adios2 filter, in function pointOrCell, is not a number");
            std::vector<std::size_t> returnBlock = {};
            return returnBlock;
        }
};

class WriterDatabases
{
    private:
        static std::map<std::string, WriterManager> m_databases;

    public:
        //WriterDatabases();
        //~WriterDatabases();

        static bool writer_exists(std::string writer_name)
        {
            auto it = m_databases.find(writer_name);
            return it != m_databases.end();
        }

        static void create_writer(std::string writer_name,
                                  std::string file_name,
                                  std::string engine_name,
                                  std::string ip_address,
                                  std::string port_number,
                                  std::string Marshal_Method)
        {
            if(writer_exists(writer_name))
                ASCENT_ERROR("Creation failed: writer already exists");
            m_databases.emplace(std::make_pair(writer_name, WriterManager(file_name, engine_name, ip_address, port_number, Marshal_Method)));
        }
        static WriterManager& get_writer(std::string writer_name)
        {
            if (!writer_exists(writer_name))
                ASCENT_ERROR("Writer database: '" << writer_name << "' does not exist");
            return m_databases[writer_name];
        }

};


class ASCENT_ADIOS2 : public ::flow::Filter
{
    public:
        ASCENT_ADIOS2();
        ~ASCENT_ADIOS2();
    private:
        virtual void   declare_interface(conduit::Node &i);
        virtual bool   verify_params(const conduit::Node &params,
                                     conduit::Node &info);
        virtual void   execute();

        std::string transportType;
        std::string fileName;
};

//-----------------------------------------------------------------------------
}; // end filters
}; // end runtime
}; // end ascent


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
