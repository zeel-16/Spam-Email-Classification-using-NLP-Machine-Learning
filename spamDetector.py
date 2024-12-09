import streamlit as st
import pickle

model = pickle.load(open('spam123.pkl','rb'))
cv = pickle.load(open('vec123.pkl','rb'))

def main():
    st.title("Spam Email Classification Application")
    st.write("This is a Machine Learning application to classify email as spam or ham.")
    st.subheader("Classification")
    user_input=st.text_area("Enter an Email to classify" , height=150)
    if st.button("Classify"):
        if user_input:
            data=[user_input]
            print(data)
            vec=cv.transform(data).toarray()
            result=model.predict(vec)
            if result[0]==0:
                st.success("This is not a Spam Email")
            else:
                st.error("This is a Spam Email")
        else:
            st.write("please enter an email to classify.")
main()