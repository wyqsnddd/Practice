# include <boost/multi_index/hashed_index.hpp>
# include <boost/multi_index_container.hpp>
# include <boost/multi_index/ordered_index.hpp>
# include <boost/multi_index/identity.hpp>
# include <boost/multi_index/member.hpp>
// # include <boost/multi_index/tag.hpp>

// # include <string>
# include <iostream>

struct mockPath
{
  mockPath(int id, int cost, int group_number, const std::string& name){ 

    id_ = id;
    cost_ = cost;

    group_number_ = group_number;
    name_ = name;
      }
  // sort by cost
  bool operator<(const mockPath& e)const{return cost_<e.cost_;}
  int id_;
  int cost_;
  int group_number_;
  std::string name_;

  
  friend std::ostream& operator<<(std::ostream& os,const mockPath& e)
  {
    os<<"ID: "<<e.id_<<" group number: "<< e.group_number_<<" name: "<<e.name_<<" cost: "<<e.cost_<<std::endl;
    return os;
  }
};


typedef boost::multi_index::multi_index_container<
  mockPath,
  boost::multi_index::indexed_by<
    // sorted by operator on cost(we use non_unique cost)
    boost::multi_index::ordered_non_unique<
      boost::multi_index::identity<mockPath>      
      >,
    // sorted by less<string> on unique name
    boost::multi_index::ordered_non_unique<
      boost::multi_index::member<mockPath, std::string, &mockPath::name_>
      >,
    boost::multi_index::hashed_unique<
      boost::multi_index::member<mockPath, int, &mockPath::id_>
      >
    >
  > path_set;

// boost::multi_index::hashed_unique<
//       tag<tags::unordered>,
//       identity<mockPath>,
//       std::hash<mockPath>


void print_out_by_name(const path_set & inputSet){
  // get a view to index 1 name
  const path_set::nth_index<1>::type& name_index=inputSet.get<1>();
  std::copy(
    name_index.begin(),name_index.end(),
    std::ostream_iterator<mockPath>(std::cout));

}

void print_out_by_cost(const path_set & inputSet){
  // get a view to index 1 name
  const path_set::nth_index<0>::type& cost_index=inputSet.get<0>();
  std::copy(
    cost_index.begin(),cost_index.end(),
    std::ostream_iterator<mockPath>(std::cout));

}



int main(){


  path_set sample_container;
  sample_container.insert(mockPath(21, 1, 2, "hello_world"));
  sample_container.insert(mockPath(21, 1, 2, "hello_world"));
  sample_container.insert(mockPath(2123, 2, 3, "bei_world"));
  sample_container.insert(mockPath(234, 2, 4, "bei_world"));
  sample_container.insert(mockPath(1243, 1, 2, "bei_Mu"));
  sample_container.insert(mockPath(8987, 1, 2, "how_are_you"));

  std::cout<<"Print out by name: "<<std::endl;
  print_out_by_name(sample_container);
  std::cout<<"Print out by cost: "<<std::endl;
  print_out_by_cost(sample_container);
  
}
