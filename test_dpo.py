from serve import CampGPTServer

server = CampGPTServer("campgpt-student-handbook")
response = server.chat("What is a 'Referral Notice' in the context of student discipline?")
print(response)