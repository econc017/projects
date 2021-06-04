fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();

    println!("s1 = {}, s2 = {}", s1, s2);
    let (s1, s2) = gives_ownership();
    println!("{} and {}", s1, s2);
}

fn gives_ownership() -> (String, String) {
    let some_string = String::from("Test");
    let some_other_string = String::from("Test2");
    return (some_string, some_other_string);
}
