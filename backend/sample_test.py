from best_model import predict_email


if __name__ == '__main__':
    # One simple test email
    sender = 'notifications@secure-mail-alert.com'
    subject = 'Verify your account immediately'
    body = 'Your account will be suspended unless you confirm your password using the link below.'
    urls = 'http://secure-login-alert.com/verify'

    prediction = predict_email(sender, subject, body, urls)

    print('\nSample Test Result')
    print('------------------')
    print('Sender :', sender)
    print('Subject:', subject)
    print('Body   :', body)
    print('URLs   :', urls)
    print('Prediction:', 'Phishing' if prediction == 1 else 'Legit')