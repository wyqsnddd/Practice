# include <boost/multi_index_container.hpp>

# include <boost/multi_index/hashed_index.hpp>
# include <boost/multi_index/ordered_index.hpp>
// Sequenced_index allows you to treat a MultiIndex container like a list of type std::list. Elements are stored in the given order.
# include <boost/multi_index/sequenced_index.hpp>


# include <boost/multi_index/identity.hpp>
# include <boost/multi_index/member.hpp>
# include <boost/multi_index/tag.hpp>

// random_access allows you to treat the MultiIndex container like a vector
#include <boost/multi_index/random_access_index.hpp>

// # include <string>
# include <iostream>

# include <boost/shared_ptr.hpp>

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
  friend std::ostream& operator<<(std::ostream& os,const mockPath* e)
  {
    os<<"ID: "<<e->id_<<" group number: "<< e->group_number_<<" name: "<<e->name_<<" cost: "<<e->cost_<<std::endl;
    return os;
  }

  
};


// A functor that is used to modify the attribute: 'id'
struct change_id{

  change_id(int id) : id_(id) {}    
  void operator()(boost::shared_ptr<mockPath> p)
  {
    p->id_ = id_;
  }

private:
  int id_;
};


// Provide us a way to access index with a name
struct tag_name {};
struct tag_id {};
struct tag_as_inserted {};

typedef boost::multi_index::multi_index_container<
  boost::shared_ptr<mockPath>,
  boost::multi_index::indexed_by<
    // (0) sorted by operator on non-unique cost(we use non_unique cost)
    boost::multi_index::ordered_non_unique<
      boost::multi_index::identity<mockPath>      
      >,
    // (1) sorted by less<string> on non-unique name
    boost::multi_index::ordered_non_unique<
      boost::multi_index::tag<tag_name>, boost::multi_index::member<mockPath, std::string, &mockPath::name_>
      >,
    // (2) hashed view
    boost::multi_index::hashed_unique<
      boost::multi_index::tag<tag_id>, boost::multi_index::member<mockPath, int, &mockPath::id_>
      >,
    // (3) as a std::list
    boost::multi_index::sequenced<
      boost::multi_index::tag<tag_as_inserted>
      >,
    // (4) access to an element with position
    boost::multi_index::random_access<>
    >
  > path_set;


void print_out_by_name(const path_set & inputSet){
  // get a view to index 1 name
  const path_set::nth_index<1>::type& name_index=inputSet.get<1>();
  std::copy(
	    name_index.begin(),name_index.end(),
	    std::ostream_iterator<boost::shared_ptr<mockPath>>(std::cout));

}

void print_out_by_cost(const path_set & inputSet){
  // get a view to index 1 name
  const path_set::nth_index<0>::type& cost_index=inputSet.get<0>();
  std::copy(
	    cost_index.begin(),cost_index.end(),
	    std::ostream_iterator<boost::shared_ptr<mockPath>>(std::cout));

}

void delete_object(int id, path_set & input_container){
  // Note that the erase operation needs to work on an index not a container. 
  path_set::index<tag_id>::type & test_id_index = input_container.get<tag_id>();
  path_set::index<tag_id>::type::iterator it = test_id_index.find(id);
  std::cout << "Found host id " << (*it)->id_ <<
    ". Attempting to delete. input_container.size() before is " << input_container.size() << " and ";
  test_id_index.erase( it ); 
  std::cout << input_container.size() << " after." << std::endl;

  
}


int main(){

  // (0) Basic insertion 
  path_set sample_container;
  boost::shared_ptr<mockPath>  newptr;
  newptr.reset(new mockPath(21, 1, 2, "hello_world"));
  sample_container.insert(newptr);
  newptr.reset(new mockPath(20, 1, 2, "hello_world"));
  sample_container.insert(newptr);
  newptr.reset(new mockPath(2123, 2, 3, "bei_world"));
  sample_container.insert(newptr);
  newptr.reset(new mockPath(234, 2, 4, "bei_world") );
  sample_container.insert(newptr);
  newptr.reset(new mockPath(1243, 1, 2, "bei_Mu") );
  sample_container.insert(newptr);
  newptr.reset(new mockPath(8987, 1, 2, "how_are_you") );
  sample_container.insert(newptr);
  newptr.reset(new mockPath(1117, 1, 2, "how_are_you") );
  sample_container.insert(newptr);
 

  // (1) iterator 
  std::cout<<"The container size is: "<<sample_container.size()<<std::endl;
  std::cout<<"Print out by name: "<<std::endl;
  print_out_by_name(sample_container);
  std::cout<<"Print out by cost: "<<std::endl;
  print_out_by_cost(sample_container);


  // (2) ordered by name
  const path_set::index<tag_name>::type & name_index = sample_container.get<tag_name>();
  if(name_index.find("test")==name_index.end())
    std::cout<<"Did not find name: test "<<std::endl;
  else
    std::cout<<"Found name: test "<<std::endl;

  // (3) ordererd by id
  // not a constant index as we want to delete and change object   
  path_set::index<tag_id>::type & id_index = sample_container.get<tag_id>();
  if(id_index.find(1243)==id_index.end())
    std::cout<<"Did not find id: 1243 "<<std::endl;
  else
    std::cout<<"Found id: 1243 "<<std::endl;
    
  std::cout<<"The amount of  id '21' is:  "<<id_index.count(21) <<std::endl;
  std::cout<<"Before deletion the container size is: "<<sample_container.size()<<std::endl;

  // (4) erase work with a certain index. 
  path_set::index<tag_id>::type::iterator it = id_index.find(21);
  id_index.erase(it);
  std::cout<<"After deletion the container size is: "<<sample_container.size()<<std::endl;

  
  it =  id_index.find(20);
  std::cout<<"Before modification, The amount of  id '20' is:  "<<id_index.count(20) <<std::endl;
  std::cout<<"Before modification, The amount of  id '7539' is:  "<<id_index.count(7593)<<std::endl;
  // (5) modification and deletion works with an index and a functor
  // (5.1) modification: 
  id_index.modify(it, change_id(7593) );
  std::cout<<"After modification, The amount of  id '20' is:  "<<id_index.count(20) <<std::endl;
  std::cout<<"After modification, The amount of  id '7539' is:  "<<id_index.count(7593) <<std::endl;  
  std::cout<<"The amount of  id '1243' is:  "<<id_index.count(1243) <<std::endl;
  std::cout<<"The amount of  id '881' is:  "<<id_index.count(881) <<std::endl;

  // (5.2) deletion:
  delete_object(234, sample_container);
  
  // (6) sequence: works as a std::list. easy to get front and end. 
  const path_set::index<tag_as_inserted>::type & inserted_index = sample_container.get<tag_as_inserted>();
  std::cout<<"The size of as_inserted is: "<<inserted_index.size()<<std::endl;
  std::cout<<"The container size is: "<<sample_container.size()<<std::endl;
  // std::cout<<"The size of as_inserted is: "<<inserted_index.()<<std::endl;
  

  // (7) random access: easy to access any element by number 
  const auto & rand_index = sample_container.get<4>();
  for (int i = 0; i < rand_index.size(); i++)
    std::cout << rand_index[i]->name_ << '\n';
  
  std::cout<<"The container size is: "<<sample_container.size()<<std::endl;
  std::cout << rand_index[0]->name_ << '\n';
  std::cout << rand_index[1]->name_ << '\n';
  std::cout << rand_index[2]->name_ << '\n';
  // sample_container.erase(rand_index[2]);
  std::cout << rand_index[3]->name_ << '\n';
  // std::cout << rand_index[4]->name_ << '\n';

  for (int i = 0; i < rand_index.size(); i++)
    std::cout << rand_index[i]->name_ << '\n';

  std::cout<<"The container size is: "<<sample_container.size()<<std::endl;

  
}
