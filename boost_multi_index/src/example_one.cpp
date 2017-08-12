# include <boost/multi_index_container.hpp>
# include <boost/multi_index/hashed_index.hpp>
# include <boost/multi_index/ordered_index.hpp>
// Sequenced_index allows you to treat a MultiIndex container like a list of type std::list. Elements are stored in the given order.
# include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/composite_key.hpp>

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
  mockPath(int id, int cost, int group_number, bool feasibility, const std::string& name){ 

    id_ = id;
    cost_ = cost;

    group_number_ = group_number;
    name_ = name;
    feasibility_ = feasibility;
  }

  // sort by cost
  bool operator<(const mockPath& e)const{return cost_<e.cost_;}
  int id_;
  int cost_;
  int group_number_;
  std::string name_;


  bool feasibility_;
  
  friend std::ostream& operator<<(std::ostream& os,const mockPath& e)
  {
    os<<"ID: "<<e.id_<<" group number: "<< e.group_number_<<" name: "<<e.name_<<" cost: "<<e.cost_<<std::endl;
    return os;
  }
  friend std::ostream& operator<<(std::ostream& os,const mockPath* e)
  {
    os<<
      "Feasibility: "<<e->feasibility_<< 
      " ID: "<<e->id_<<" group number: "<< e->group_number_<<" name: "<<e->name_<<" cost: "<<e->cost_<<std::endl;
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
struct tag_composite {};

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
    boost::multi_index::random_access<>,
    // (5) count on the bool: feasibility so we use hash_non_unique instead of hash_unique
    // If we count on words, we use orderd_non_unique<>.
    boost::multi_index::hashed_non_unique<
      boost::multi_index::member<mockPath, bool, &mockPath::feasibility_>
      >,
    // (6) We use composite keys to sort two keys in a lexicographical order: (1) feasibility (2) cost
    boost::multi_index::ordered_non_unique<
      boost::multi_index::tag<tag_composite>, boost::multi_index::composite_key<
	mockPath,
	boost::multi_index::member<mockPath, bool, &mockPath::feasibility_>,
	boost::multi_index::identity<mockPath>      
	>
      >
    >// end of indexed_by
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

void print_out_composite_key(const path_set & inputSet){
  // get a view to index 1 name
  const path_set::index<tag_composite>::type & composite_index = inputSet.get<tag_composite>();
  std::copy(
	    composite_index.begin(),composite_index.end(),
	    std::ostream_iterator<boost::shared_ptr<mockPath>>(std::cout));

}

void delete_object(int id, path_set & input_container){
  // Note that the erase operation needs to work on an index not a container. 
  path_set::index<tag_id>::type & test_id_index = input_container.get<tag_id>();

  int amount = test_id_index.count(id);
  if (amount){
    std::cout<<"The amount of  id: "<<id<< " is:  "<<amount <<std::endl;  
    return;
  }
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
  newptr.reset(new mockPath(21, 1, 2, true, "hello_world"));
  sample_container.insert(newptr);
  newptr.reset(new mockPath(20, 1, 2, false, "hello_world"));
  sample_container.insert(newptr);
  newptr.reset(new mockPath(2123, 2, 3, true, "bei_world"));
  sample_container.insert(newptr);
  newptr.reset(new mockPath(234, 2, 4, false, "bei_world") );
  sample_container.insert(newptr);
  newptr.reset(new mockPath(1243, 1, 2, true,  "bei_Mu") );
  sample_container.insert(newptr);
  newptr.reset(new mockPath(8987, 1, 2, false, "how_are_you") );
  sample_container.insert(newptr);
  newptr.reset(new mockPath(1117, 1, 2, true,  "how_are_you") );
  sample_container.insert(newptr);
 

  // (1) iterator 
  std::cout<<"The container size is: "<<sample_container.size()<<std::endl;
  std::cout<<"Print out by name: "<<std::endl;
  print_out_by_name(sample_container);
  std::cout<<"Print out by cost: "<<std::endl;
  print_out_by_cost(sample_container);

  const path_set::nth_index<0>::type & test_cost_index = sample_container.get<0>();

  /// Note that this 'count' operation has logarithmic complexity
  std::cout<<"Number of path with a cost equals to 2 is: "<<test_cost_index.count( mockPath(1117, 1, 2, false, "how_are_you") ) <<std::endl;
  
  
  

  // (2) ordered by name
  const path_set::index<tag_name>::type & name_index = sample_container.get<tag_name>();
  if(name_index.find("test")==name_index.end())
    std::cout<<"Did not find name: test "<<std::endl;
  else
    std::cout<<"Found name: test "<<std::endl;

  std::cout<<"Number of path with a name 'hello_world' is: "<<name_index.count( "hello_world" ) <<std::endl;

  
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
  newptr.reset(new mockPath(64, 2, 4, false, "bei_world") );
  std::cout<<"Before insertion the container size is: "<<sample_container.size()<<std::endl;
  sample_container.insert(newptr);
  std::cout<<"Inserted path with id: 64"<<std::endl;
  std::cout<<"After insertion the container size is: "<<sample_container.size()<<std::endl;
  print_out_by_name(sample_container);

  
  delete_object(64, sample_container);
  
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
  
  // (8) Count on the feasible path
  const path_set::nth_index<5>::type& feasibility_index=sample_container.get<5>();
  std::cout<<"The number of feasible path is: "<<feasibility_index.count(true)<<std::endl;
  std::cout<<"The number of infeasible path is: "<<feasibility_index.count(false)<<std::endl;

  print_out_by_name(sample_container);
  // (9) print out by the composite key
  std::cout<<std::endl<<" Container sorted by the composite key (1) feasibility (2) cost"<<std::endl;
  print_out_composite_key(sample_container);  
  
}
