main() {
  correct_count_of GIVE
  bic_norm | flatten_words | grep -vF "*" | count | trim | while read entry; do
    NUMBER=$(echo $entry | cut -d ' ' -f1)
    WORD=$(echo $entry | cut -d ' ' -f2 | tr -d '*')
    echo $NUMBER/$(correct_count_of $WORD) $WORD
  done
}

correct_count_of() {
  correct_count | trim | grep ${1}$ | cut -d ' ' -f1
}

trim() {
  sed -E 's/^ +//'
}

correct_count() {
  correct | flatten_words | count
}

count() {
  sort | uniq -c | sort -n
}

flatten_words() {
  while read line; do for word in $line; do echo $word; done; done
}

bic_norm() {
  cat <<EOF
  HN WRITE *ARRIVE                                            
  ARY *NEW GO *WHAT                                           
  MARY *HAVE *GO1 CAN                                         
  MARY *BOX *HAVE *GO *CAR *CAR *CHICKEN *WRITE               
  MARY LIKE *MARY *LIKE *MARY                                 
  ANN *ANN *ANN *ANN *ANN                                     
  IX-1P *IX *MARY IX IX                                       
  ARY *MARY *YESTERDAY *SHOOT LIKE *IX                        
  MARY *JOHN *FUTURE1 *VEGETABLE *MARY                        
  OHN *FUTURE BUY HOUSE                                       
  POSS *SEE *WRITE CAR *HAVE                                  
  OHN *FUTURE *FUTURE *STUDENT HOUSE                          
  IX *IX *IX MARY                                             
  MARY *IX *JOHN *ARRIVE HOUSE                                
  OHN WILL VISIT MARY                                         
  IX *BILL VISIT MARY                                         
  JOHN BLAME MARY                                             
  JOHN *HAVE *VISIT BOOK                                      
  FUTURE *THROW *IX *IX IX *ARRIVE *BREAK-DOWN                
  *SELF *IX IX *IX WOMAN *CHOCOLATE                            
  OHN *MAN IX *IX *IX BOOK                                    
  POSS NEW CAR BREAK-DOWN                                     
  JOHN *POSS                                                  
  *MARY POSS *BOX *MARY *TOY1                                 
  *IX *HOMEWORK                                               
  IX CAR *IX *JOHN *BOX                                       
  SUE *BUY1 IX CAR *FINISH                                    
  JOHN *GIVE1 BOOK                                            
  JOHN *BUY1 *CAR YESTERDAY BOOK                              
  JOHN BUY YESTERDAY WHAT BOOK                                
  LOVE *IX WHO                                                
  *MARY IX *SAY-1P LOVE *IX                                   
  *MARY *IX BLAME                                             
  *NEW *GIVE1 GIVE1 *VISIT *CAR                               
  JOHN *BOX                                                   
  *IX BOY *GIVE1 TEACHER APPLE                                
  *JANA *MARY *PREFER *ARRIVE                                 
  *IX *YESTERDAY *PREFER BOX                                  
  *JOHN CHOCOLATE *JOHN                                       
  JOHN *GIVE1 *IX *WOMAN *STUDENT HOUSE                       
EOF
}

correct() {
  cat << EOF
  JOHN WRITE HOMEWORK
  JOHN CAN GO CAN
  JOHN CAN GO CAN
  JOHN FISH WONT EAT BUT CAN EAT CHICKEN
  JOHN LIKE IX IX IX
  JOHN LIKE IX IX IX
  JOHN LIKE IX IX IX
  MARY VEGETABLE KNOW IX LIKE CORN1
  JOHN IX THINK MARY LOVE
  JOHN MUST BUY HOUSE
  FUTURE JOHN BUY CAR SHOULD
  JOHN SHOULD NOT BUY HOUSE
  JOHN DECIDE VISIT MARY
  JOHN FUTURE NOT BUY HOUSE
  JOHN WILL VISIT MARY
  JOHN NOT VISIT MARY
  ANN BLAME MARY
  IX-1P FIND SOMETHING-ONE BOOK
  JOHN IX GIVE MAN IX NEW COAT
  JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
  JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
  POSS NEW CAR BREAK-DOWN
  JOHN LEG
  JOHN POSS FRIEND HAVE CANDY
  WOMAN ARRIVE
  IX CAR BLUE SUE BUY
  SUE BUY IX CAR BLUE
  JOHN READ BOOK
  JOHN BUY WHAT YESTERDAY BOOK
  JOHN BUY YESTERDAY WHAT BOOK
  LOVE JOHN WHO
  JOHN IX SAY LOVE MARY
  JOHN MARY BLAME
  PEOPLE GROUP GIVE1 JANA TOY
  JOHN ARRIVE
  ALL BOY GIVE TEACHER APPLE
  JOHN GIVE GIRL BOX
  JOHN GIVE GIRL BOX
  LIKE CHOCOLATE WHO
  JOHN TELL MARY IX-1P BUY HOUSE
EOF
}

main
