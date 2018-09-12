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

/******************************************************
TODO:

 */


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_adios2_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_adios2_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_file_system.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi related includes

#ifdef ADIOS2_HAVE_MPI
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif

#include <set>
//#include <cstring>
#include <limits>
#include <cmath>
#include <string>
#include <cstddef> // std::size_t
#include <cinttypes> // std::int64_t

//static adios2::Engine adiosWriter = NULL;

using namespace std;
using namespace conduit;
using namespace flow;

struct coordInfo
{
    coordInfo(int r, int n, double r0, double r1) : num(n), rank(r) {range[0]=r0; range[1]=r1;}
    coordInfo() {num=0; rank=-1; range[0]=range[1]=0;}
    coordInfo(const coordInfo &c) {num=c.num; rank=c.rank; range[0]=c.range[0]; range[1]=c.range[1];}

    int num, rank;
    double range[2];
};

inline bool operator<(const coordInfo &c1, const coordInfo &c2)
{
    return c1.range[0] < c2.range[0];
}

inline ostream& operator<<(ostream &os, const coordInfo &ci)
{
    os<<"(r= "<<ci.rank<<" : n= "<<ci.num<<" ["<<ci.range[0]<<","<<ci.range[1]<<"])";
    return os;
}

template <class T>
inline std::ostream& operator<<(ostream& os, const vector<T>& v)
{
    os<<"[";
    auto it = v.begin();
    for ( ; it != v.end(); ++it)
        os<<" "<< *it;
    os<<"]";
    return os;
}

template <class T>
inline ostream& operator<<(ostream& os, const set<T>& s)
{
    os<<"{";
    auto it = s.begin();
    for ( ; it != s.end(); ++it)
        os<<" "<< *it;
    os<<"}";
    return os;
}

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
//#ifdef ADIOS2_HAVE_MPI
WriterManager::WriterManager(std::string file_name,         // Name of our file
                             std::string engine_name,       // Name of our engine
                             std::string ip_address,        // IP Address for DataMan (default 127.0.0.1)
                             std::string port_number,       // Port for DataMan (default 20408)
                             std::string Marshal_Method)    // Marshal method for SST (default FFS)
    :
#ifdef ADIOS2_HAVE_MPI
     mpi_comm(MPI_Comm_f2c(Workspace::default_mpi_comm())),
     adios(std::make_shared<adios2::ADIOS>(mpi_comm, adios2::DebugON)),
#else
     adios(std::make_shared<adios2::ADIOS>(adios2::DebugON)),
#endif
     m_IO(adios -> DeclareIO(file_name)),
     m_fileName(file_name),
     m_engineName(engine_name),
     m_ip(ip_address),
     m_port(port_number),
     m_marshal(Marshal_Method)
{
#ifndef ADIOS2_HAVE_MPI
    rank = 0;
    mpi_comm = 0;
#endif
    m_meshName = "mesh";
    m_globalDims.resize(4);
    m_localDims.resize(4);
    m_offset.resize(4);
    for ( std::size_t i = 0; i < 4; ++i )
        m_globalDims[i] = m_localDims[i] = m_offset[i] = 0;

#ifdef ADIOS2_HAVE_MPI
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &numRanks);
#endif
    m_globalDims[0] = numRanks;
    m_localDims[0] = 1;
    //m_offset[0] = rank;
    m_explicitMesh = false;

    if ( m_engineName == "sst" )
    {
        m_IO.SetParameter("MarshalMethod", m_marshal);
    }
    else if ( m_engineName == "dataman" )
    {
        //m_IO.SetParameter("WorkflowMode", "subscribe"); // Subscribe doesn't work with our workflow
        m_IO.SetParameter("WorkflowMode", "p2p");   // p2p only supported now
        // We restrict to WAN and ZMQ
        m_IO.AddTransport("WAN", {{"Library", "ZMQ"},{"IPAddress", m_ip}, {"Port", m_port}});
    }

    // Let's set the engine after parameters have been set
    m_IO.SetEngine(m_engineName);   // Take any name given by "transport"
#ifdef ADIOS2_HAVE_MPI
    m_adiosWriter = m_IO.Open(m_fileName, adios2::Mode::Write, mpi_comm);
#else
    m_adiosWriter = m_IO.Open(m_fileName, adios2::Mode::Write);
#endif
} // End WriterManager 

WriterManager::WriterManager()
    :adios(std::make_shared<adios2::ADIOS>(adios2::DebugON)),
     m_IO(adios -> DeclareIO("adios2")),
     m_fileName("adios2.bp"),
     m_engineName("BPFile")
{
    mpi_comm = 0;
    rank = 0;
    numRanks = 1;
    m_meshName = "mesh";
    m_globalDims.resize(4);
    m_localDims.resize(4);
    m_offset.resize(4);
    for ( std::size_t i = 0; i < 4; ++i )
        m_globalDims[i] = m_localDims[i] = m_offset[i] = 0;

    m_globalDims[0] = numRanks;
    m_localDims[0] = 1;
    m_offset[0] = rank;
    m_explicitMesh = false;
    
    m_fileName = "adios2.bp";
    m_IO.SetEngine("BPFile");

    m_adiosWriter = m_IO.Open(m_fileName, adios2::Mode::Write);
}

// Define as a global object
std::map<std::string, WriterManager>WriterDatabases::m_databases;


WriterManager::~WriterManager()
{
    // Files will close themselves when IO goes out of scope
    //m_adiosWriter.Close();
}


//-----------------------------------------------------------------------------
ASCENT_ADIOS2::ASCENT_ADIOS2()
{}
ASCENT_ADIOS2::~ASCENT_ADIOS2()
{}

//-----------------------------------------------------------------------------
void
ASCENT_ADIOS2::declare_interface(Node &i)
{
    i["type_name"]   = "adios2";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
ASCENT_ADIOS2::verify_params(const conduit::Node &params,
                     conduit::Node &info)
{
    bool res = true;

    if (!params.has_child("transport") ||
        !params["transport"].dtype().is_string())
    {
        info["errors"].append() = "missing required entry 'transport'";
        res = false;
    }


    if (!params.has_child("filename") ||
        !params["transport"].dtype().is_string() )
    {
        info["errors"].append() = "missing required entry 'filename'";
        res = false;
    }


    return res;
}

//-----------------------------------------------------------------------------
void
ASCENT_ADIOS2::execute()
{

    ASCENT_INFO("execute");

    if(!input("in").check_type<Node>())
    {
        // error
        ASCENT_ERROR("adios2 filter requires a conduit::Node input");
    }

    transportType = params()["transport"].as_string();
    fileName      = params()["filename"].as_string();
    vector<string> variables;

    if (params().has_child("variables"))
    {
        string str = params()["variables"].as_string();
        stringstream ss(str);
        string s;
        while (ss >> s)
        {
            if (s.size() == 0)
                continue;
            if (s[s.size()-1] == ',')
                s = s.substr(0, s.size()-1);
            variables.push_back(s);
            ss.ignore(1);
        }
    }
    std::string writer_name;
    if (params().has_child("writer"))   // Multiple writers can be made
        writer_name = params()["writer"].as_string();
    else                                // Only one writer
        writer_name = params()["transport"].as_string();
        //writer_name = "default_writer";
    std::string ip_address = "127.0.0.1";
    std::string port_number = "20408";
    if ( params().has_child("transport") && params()["transport"].as_string() == "dataman" )
    {
        if (params().has_child("ip"))
            ip_address = params()["ip"].as_string();
        if (params().has_child("port"))
            port_number = params()["port"].as_string();
        std::cout << "Setting IP address to: " << ip_address << std::endl;
    }
    std::string Marshal_Method = "FFS";
    if ( params().has_child("marshal") )
        Marshal_Method = params()["marshal"].as_string();


    if (!WriterDatabases::writer_exists(writer_name))
        WriterDatabases::create_writer(writer_name,
                                       fileName,
                                       transportType,
                                       ip_address,
                                       port_number,
                                       Marshal_Method);
    WriterManager &manager = WriterDatabases::get_writer(writer_name);

    manager.m_adiosWriter.BeginStep();

    //Fetch input data
    Node *blueprint_data = input<Node>("in");
    Node dom = blueprint_data -> child(0);

    NodeConstIterator itr = dom["coordsets"].children();
    while (itr.has_next())
    {

        const Node &coordSet = itr.next();
        std::string coordSetType = coordSet["type"].as_string();

        if (coordSetType == "uniform")
        {
            // Unsupported 
            manager.UniformMeshSchema(coordSet);
            break;
        }
        else if (coordSetType == "rectilinear")
        {
            manager.RectilinearMeshSchema(coordSet); 
            break;
        }
        else if (coordSetType == "explicit")
        {
            manager.ExplicitMeshSchema(coordSet);
            //manager.m_explicitMesh = true;
            break;
        }
        else
        {
        	cout<<"***************************************"<<endl;
        	cout<<"*****WARNING: meshType("<<coordSetType<<") not yet supported"<<endl;
        }
    }

    if (dom.has_child("fields"))
    {

        // if we don't specify a topology, find the first topology ...
        NodeConstIterator itr = dom["topologies"].children();
        itr.next();
        std::string  topo_name = itr.name();

        // as long as mesh blueprint verify true, we access data without fear.
        const Node &n_topo   = dom["topologies"][topo_name];
        const Node &fields = dom["fields"];
        NodeConstIterator fields_itr = fields.children();

        // For each of the field variables
        while(fields_itr.has_next())
        {
            const Node& field = fields_itr.next();
            std::string field_name = fields_itr.name();
            bool saveField = (variables.empty() ? true : false);

            for (auto &v : variables)
                if (field_name == v)
                {
                    saveField = true;
                    break;
                }

           if (saveField)
                manager.FieldVariable(field_name, field, n_topo);
        }
        //manager.m_fieldVarFirst = false;
    }
    manager.m_fieldVarFirst = false;

    manager.m_adiosWriter.EndStep();
    if ( manager.m_engineName == "hdf5" )
        manager.m_adiosWriter.Flush();
}

//-----------------------------------------------------------------------------
bool
WriterManager::UniformMeshSchema(const Node &node)
{
    if (rank == 0)
    {
        cout<<"***************************************"<<endl;
        cout<<"*****WARNING: ADIOS2 Uniform mesh schema not yet supported "<<endl;
    }
    return false;
}


//-----------------------------------------------------------------------------
bool
WriterManager::CalcExplicitMeshInfo(const conduit::Node &node, vector<vector<double>> &XYZ)
{
    const Node &X = node["values/x"];
    const Node &Y = node["values/y"];
    const Node &Z = node["values/z"];
    //const Node &topo = node["connectivity"];
    //const double *connectivityPtr = topo.as_float64_ptr();

    const double *xyzPtr[3] = {X.as_float64_ptr(),
                               Y.as_float64_ptr(),
                               Z.as_float64_ptr()};

    m_localDims = {X.dtype().number_of_elements(),
                 Y.dtype().number_of_elements(),
                 Z.dtype().number_of_elements(),
                 0};

    cout<<"***************************************"<<endl;
    cout<<"*****x size: "<<X.dtype().number_of_elements()<<endl;
    cout<<"*****y size: "<<Y.dtype().number_of_elements()<<endl;
    cout<<"*****z size: "<<Z.dtype().number_of_elements()<<endl;

    //Stuff the XYZ coordinates into the conveniently provided array.
    XYZ.resize(3);
    for (int i = 0; i < 3; i++)
    {
        XYZ[i].resize(m_localDims[i]);
        memcpy(&(XYZ[i][0]), xyzPtr[i], m_localDims[i]*sizeof(double));
    }

    //Participation trophy if you only bring 1 rank to the game.
    if (numRanks == 1)
    {
        m_offset = {0,0,0,0};
        m_globalDims = m_localDims;
        return true;
    }

#ifdef ADIOS2_HAVE_MPI

    // Have to figure out the indexing for each rank.
    vector<int> ldims(3*numRanks, 0), buff(3*numRanks,0);
    ldims[3*rank + 0] = m_localDims[0];
    ldims[3*rank + 1] = m_localDims[1];
    ldims[3*rank + 2] = m_localDims[2];

    int mpiStatus;
    mpiStatus = MPI_Allreduce(&ldims[0], &buff[0], ldims.size(), MPI_INT, MPI_SUM, mpi_comm);
    if (mpiStatus != MPI_SUCCESS)
        return false;

    //Calculate the global dims. This is just the sum of all the localDims.
    m_globalDims = {0,0,0};
    for (int i = 0; i < buff.size(); i+=3)
    {
        m_globalDims[0] += buff[i + 0];
        m_globalDims[1] += buff[i + 1];
        m_globalDims[2] += buff[i + 2];
    }

    //And now for the offsets. It is the sum of all the localDims before me.
    m_offset[0] = 0; //with more than one rank, this has to be reset to 0 now
    for (int i = 0; i < rank; i++)
    {
        m_offset[0] += buff[i*3 + 0];
        m_offset[1] += buff[i*3 + 1];
        m_offset[2] += buff[i*3 + 2];
    }
    return true;

#endif
}


//-----------------------------------------------------------------------------
bool
WriterManager::ExplicitMeshSchema(const Node &node)
{
    if (!node.has_child("values"))
        return false;

    const Node &coords = node["values"];
    if (!coords.has_child("x") || !coords.has_child("y") || !coords.has_child("z"))
        return false;

    vector<vector<double>> XYZ;
    if (!CalcExplicitMeshInfo(node, XYZ))
        return false;

    std::vector<std::string> coordNames = {"coords_x", "coords_y", "coords_z"};

    std::vector<adios2::Variable<double>> myVars;
    //Write schema metadata for Expl. Mesh.
    if (m_explicitMeshFirst)
    {
        cout<<"**************************************************"<<endl;
        cout << rank << ": globalDims: " << pointOrCell(m_globalDims) << endl;
        cout << rank << ": localDims:  " << pointOrCell(m_localDims)  << endl;
        cout << rank << ": offset:     " << pointOrCell(m_offset)     << endl;

        std::cout << "Writing attributes" << std::endl;
        m_IO.DefineAttribute<std::string>("adios2_schema/version_major",
                                            std::to_string(ADIOS2_VERSION_MAJOR));
        m_IO.DefineAttribute<std::string>("adios2_schema/version_minor",
                                            std::to_string(ADIOS2_VERSION_MINOR));
        m_IO.DefineAttribute<std::string>("/adios2_schema/mesh/type", 
                                             "explicit");
        m_IO.DefineAttribute<std::int64_t>("adios2_schema/mesh/dimension0",
                                             m_globalDims[0]);
        m_IO.DefineAttribute<std::int64_t>("adios2_schema/mesh/dimension1",
                                             m_globalDims[1]);
        m_IO.DefineAttribute<std::int64_t>("adios2_schema/mesh/dimension2",
                                             m_globalDims[2]);
        m_IO.DefineAttribute<std::int64_t>("adios2_schema/mesh/dimension3",
                                             m_globalDims[3]);
        m_IO.DefineAttribute<std::int64_t>("adios2_schema/mesh/dimension-num",
                                             m_globalDims.size());
        for ( std::size_t i = 0; i < coordNames.size(); ++i )
        {
            m_IO.DefineAttribute<std::string>("adios2_schema/mesh/coords-multi-var" + std::to_string(i),
                                                 coordNames[i]);
        }
        m_IO.DefineAttribute<std::size_t>("adios2_schema/mesh/coords-multi-var-num",
                                             coordNames.size());

        //Write out coordinates.
        for (std::size_t i = 0; i < coordNames.size(); ++i)
        {
            std::vector<std::size_t> g = {std::size_t(numRanks), std::size_t(m_globalDims[i])};
            std::vector<std::size_t> l = {1, std::size_t(m_localDims[i])};
            std::vector<std::size_t> o = {std::size_t(rank), std::size_t(m_offset[i])};
            myVars.emplace_back(
                    m_IO.DefineVariable<double>(coordNames[i],
                                                pointOrCell(g),
                                                pointOrCell(o),
                                                pointOrCell(l)));
        }
        m_explicitMeshFirst = false;
    }
    // Add all the Coordinate data to the IO step
    for ( std::size_t i = 0; i < coordNames.size(); ++i )
        m_adiosWriter.Put<double>(myVars[i], XYZ[i].data());
    m_adiosWriter.PerformPuts(); // Similar to Mode::Sync. 
    m_explicitMesh = true;
    return true;
}


//-----------------------------------------------------------------------------
bool
WriterManager::CalcRectilinearMeshInfo(const conduit::Node &node,
                               vector<vector<double>> &XYZ)
{
    const Node &X = node["x"];
    const Node &Y = node["y"];
    const Node &Z = node["z"];

    const double *xyzPtr[3] = {X.as_float64_ptr(),
                               Y.as_float64_ptr(),
                               Z.as_float64_ptr()};

    m_localDims = {1,
                 X.dtype().number_of_elements(),
                 Y.dtype().number_of_elements(),
                 Z.dtype().number_of_elements()};

    //Stuff the XYZ coordinates into the conveniently provided array.
    XYZ.resize(3);
    for (int i = 0; i < 3; i++)
    {
        XYZ[i].resize(m_localDims[i+1]);
        memcpy(&(XYZ[i][0]), xyzPtr[i], m_localDims[i+1]*sizeof(double));
    }

    //Participation trophy if you only bring 1 rank to the game.
    if (numRanks == 1)
    {
        m_offset = {0,0,0,0};
        m_globalDims = m_localDims;
        return true;
    }

#ifdef ADIOS2_HAVE_MPI

    // Have to figure out the indexing for each rank.
    vector<int> ldims(3*numRanks, 0); //, buff(3*numRanks,0);
    ldims[3*rank + 0] = m_localDims[0];
    ldims[3*rank + 1] = m_localDims[1];
    ldims[3*rank + 2] = m_localDims[2];

    m_globalDims = {numRanks,m_localDims[1],m_localDims[2],m_localDims[3]};
    m_offset = {rank,0,0,0};

    return true;

#endif
}


//-----------------------------------------------------------------------------
bool
WriterManager::RectilinearMeshSchema(const Node &node)
{
    if (!node.has_child("values"))
        return false;

    const Node &coords = node["values"];
    if (!coords.has_child("x") || !coords.has_child("y") || !coords.has_child("z"))
        return false;

    vector<vector<double>> XYZ;
    if (!CalcRectilinearMeshInfo(coords, XYZ))
        return false;

    const std::vector<std::string> coordNames{"coords_x", "coords_y", "coords_z"};

    //Write schema metadata for Rect Mesh. But only if first time
    if (m_recFirst)
    {
        //cout<<"**************************************************"<<endl;
        //cout << "rank: " << rank << ": globalDims: " << pointOrCell(globalDims) << endl;
        //cout << "rank: " << rank << ": localDims:  " << pointOrCell(localDims)  << endl;
        //cout << "rank: " << rank << ": offset:     " << pointOrCell(offset)     << endl;

        std::cout << "Writing attributes" << std::endl;
        m_IO.DefineAttribute<std::string>("adios2_schema/version_major",
                                            std::to_string(ADIOS2_VERSION_MAJOR));
        m_IO.DefineAttribute<std::string>("adios2_schema/version_minor",
                                            std::to_string(ADIOS2_VERSION_MINOR));
        m_IO.DefineAttribute<std::string>("/adios2_schema/mesh/type", 
                                             "rectilinear");
        m_IO.DefineAttribute<std::size_t>("adios2_schema/mesh/dimension0",
                                             m_globalDims[0]);
        m_IO.DefineAttribute<std::size_t>("adios2_schema/mesh/dimension1",
                                             m_globalDims[1]);
        m_IO.DefineAttribute<std::size_t>("adios2_schema/mesh/dimension2",
                                             m_globalDims[2]);
        m_IO.DefineAttribute<std::size_t>("adios2_schema/mesh/dimension3",
                                             m_globalDims[3]);
        m_IO.DefineAttribute<std::size_t>("adios2_schema/mesh/dimension-num",
                                             m_globalDims.size());
        for ( std::size_t i = 0; i < coordNames.size(); ++i )
        {
            m_IO.DefineAttribute<std::string>("adios2_schema/mesh/coords-multi-var" + std::to_string(i),
                                                 coordNames[i]);
        }
        m_IO.DefineAttribute<std::size_t>("adios2_schema/mesh/coords-multi-var-num",
                                             coordNames.size());
        for ( std::size_t i = 0; i < coordNames.size(); ++i )
        {
            std::vector<std::size_t> g = {std::size_t(numRanks), std::size_t(m_globalDims[i+1])};
            std::vector<std::size_t> l = {1, std::size_t(m_localDims[i+1])};
            std::vector<std::size_t> o = {std::size_t(rank), std::size_t(m_offset[i+1])};
            myVars.emplace_back(
                    m_IO.DefineVariable<double>(coordNames[i],
                                                pointOrCell(g),
                                                pointOrCell(o),
                                                pointOrCell(l)));
        }
        m_recFirst = false;
    }
    for ( std::size_t i = 0; i < coordNames.size(); ++i )
        m_adiosWriter.Put<double>(myVars[i], XYZ[i].data());
    m_adiosWriter.PerformPuts();  // Acts like Mode::Sync

    return true;
}

//-----------------------------------------------------------------------------
bool
WriterManager::FieldVariable(const string &fieldName, const Node &node, const Node &n_topo)
{
    // TODO: we can assume this is true if verify is true and this is a rect mesh.
    if (!node.has_child("values") ||
        !node.has_child("association") ||
        !node.has_child("type"))
    {
        cerr << "Field Variable not supported at this time" << endl;
        return false;
    }

    const string &fieldType = node["type"].as_string();
    const string &fieldAssoc = node["association"].as_string();

    if (fieldType != "scalar")
    {
        ASCENT_INFO("Field type "
                    << fieldType
                    << " not supported for ASCENT_ADIOS2 this time");
        cerr << "Field type " << fieldType << " not supported for ASCENT_ADIOS2 at this time";
        return false;
    }
    if (fieldAssoc != "vertex" && fieldAssoc != "element")
    {
        ASCENT_INFO("Field association "
                    << fieldAssoc
                    <<" not supported for ASCENT_ADIOS2 this time");
        cerr << "Field association " << fieldAssoc << " not supported for ASCENT_ADIOS2 at this time";
        return false;
    }

    const Node &field_values = node["values"];
    
    if (m_fieldVarFirst) 
    {
        m_IO.DefineAttribute<std::string>(fieldName + "/adios2_schema",
                                          "mesh"); // Check that always mesh
        m_IO.DefineAttribute<std::string>(fieldName + "/adios2_schema/centering",
                                          (fieldAssoc == "vertex") ? "point" : "cell");
    }

    if(!m_explicitMesh)
    {
        if (m_fieldVarFirst)
        {
            
            m_fieldVariableMap[fieldName] = 
                m_IO.DefineVariable<double>(fieldName,
                                            pointOrCell(m_globalDims, (fieldAssoc=="vertex")),
                                            pointOrCell(m_offset, (fieldAssoc=="vertex")),
                                            pointOrCell(m_localDims, (fieldAssoc=="vertex")));
        }
        std::vector<double> tmpField(field_values.dtype().number_of_elements());
        memcpy(&tmpField[0], field_values.as_double_ptr(), field_values.dtype().number_of_elements()*sizeof(double));
        if(m_adiosWriter)
        {
            m_adiosWriter.Put<double>(m_fieldVariableMap[fieldName], field_values.as_double_ptr());
            m_adiosWriter.PerformPuts();
        }
        else
            std::cout << "Engine adiosWriter is not valid\n";
    }
    else //need to write seperate array for each cell type
    {
        string mesh_type = n_topo["type"].as_string();

        n_topo.print_detailed();
        cerr << "mesh type = " << mesh_type << endl;

        if(mesh_type == "structured") //assuming we have a cube here, needs expanded for zoo
        {
            //const Node &n_topo_eles = n_topo["elements"];
            int numElements = field_values.dtype().number_of_elements();

            std::vector<int64_t> globalDimsCopy, localDimsCopy;
            if(!m_explicitMesh)
            {
                globalDimsCopy.resize(4);
                localDimsCopy.resize(4);
                globalDimsCopy = m_globalDims;
                localDimsCopy = m_localDims;
            }
            else
            {
                globalDimsCopy.resize(3);
                localDimsCopy.resize(3);
                globalDimsCopy = {0, 0, 0};
                localDimsCopy = {0, 0, 0};
                for(int i = 0; i < 3; i++)
                {
                    globalDimsCopy[i] = std::pow(m_globalDims[i], 1/3.);
                    localDimsCopy[i] = std::pow(m_localDims[i], 1/3.);
                }
            }
        }
    }
    return true;
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
